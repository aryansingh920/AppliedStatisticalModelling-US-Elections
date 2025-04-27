#!/usr/bin/env Rscript

# --- Set CRAN mirror & install required packages if missing ---
options(repos="https://cloud.r-project.org")
required_packages <- c("mgcv", "readr", "ggplot2", "sf", "viridis", "gridExtra", "knitr", "rmarkdown")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly=TRUE)) install.packages(pkg)
}

# Load packages
library(mgcv)
library(readr)
library(ggplot2)
library(sf)
library(viridis)
library(gridExtra)

# --- 1. Read and prepare data ---
# Make sure the file path is correct
uv <- read_csv("US Votes Data.csv", show_col_types=FALSE)

# Make proportions
uv$per_gop_prop <- uv$per_gop / 100
uv$depr_prop <- uv$Crude.Prevalence.Estimate / 100
uv$white_prop <- uv$race

# Create binary response (majority Republican vote)
uv$rep_majority <- ifelse(uv$per_gop > 50, 1, 0)

# Calculate GOP votes from percentage and total votes
uv$GOP_votes <- round(uv$per_gop_prop * uv$total_votes)
uv$DEM_votes <- uv$total_votes - uv$GOP_votes

# Ensure states are a factor
uv$STNAME <- as.factor(uv$STNAME)

# Standardize continuous predictors for effect size comparison
uv$depr_prop_std <- scale(uv$depr_prop)
uv$white_prop_std <- scale(uv$white_prop)

# Check for additional covariates
potential_covariates <- c("median_income", "college_education", "unemployment", "rural_percentage")
available_covariates <- potential_covariates[potential_covariates %in% names(uv)]

if(length(available_covariates) > 0) {
  # Standardize available covariates
  for(cov in available_covariates) {
    uv[[paste0(cov, "_std")]] <- scale(uv[[cov]])
  }
}

# --- 2. Fit models ---
# Base model with proper binomial likelihood on counts
# Add available covariates if they exist
if(length(available_covariates) > 0) {
  covariates_formula <- paste(paste0(available_covariates, "_std"), collapse=" + ")
  base_formula <- as.formula(paste("cbind(GOP_votes, DEM_votes) ~ depr_prop_std + white_prop_std +", 
                                   covariates_formula, "+ s(STNAME, bs='re')"))
} else {
  base_formula <- cbind(GOP_votes, DEM_votes) ~ depr_prop_std + white_prop_std + s(STNAME, bs="re")
}

m_base2 <- bam(
  base_formula,
  data = uv,
  family = binomial(link="logit"),
  discrete = TRUE
)

# Fixed effects model without random effects (for comparison)
if(length(available_covariates) > 0) {
  fixed_formula <- as.formula(paste("cbind(GOP_votes, DEM_votes) ~ depr_prop_std + white_prop_std +", 
                                   covariates_formula))
} else {
  fixed_formula <- cbind(GOP_votes, DEM_votes) ~ depr_prop_std + white_prop_std
}

m_fixed2 <- bam(
  fixed_formula,
  data = uv,
  family = binomial(link="logit")
  # No need for discrete=TRUE when no smooths
)

# Check model for adequate k in smooth terms
# Start with a higher k value for depression effect
m_smooth_depr2 <- bam(
  cbind(GOP_votes, DEM_votes) ~ s(depr_prop_std, k=15) + white_prop_std + s(STNAME, bs="re"),
  data = uv,
  family = binomial(link="logit"),
  discrete = TRUE
)

# Check if k=15 is adequate
gam_check_smooth <- gam.check(m_smooth_depr2)
# If k still needs adjustment, we'll check the output later

# Random slopes model with proper binomial likelihood
m_rand_slope2 <- bam(
  cbind(GOP_votes, DEM_votes) ~ white_prop_std + 
    s(STNAME, bs="re") +                      # Random intercept
    s(STNAME, by=depr_prop_std, bs="re"),     # Random slope
  data = uv,
  family = binomial(link="logit"),
  discrete = TRUE
)

# Weighted binary outcome model (GOP majority)
m_binary2 <- bam(
  rep_majority ~ s(depr_prop_std, k=15) + white_prop_std + s(STNAME, bs="re"),
  data = uv,
  family = binomial(link="logit"),
  weights = total_votes,  # Weight by county size
  discrete = TRUE
)

# Check if k=15 is adequate for binary model
gam_check_binary <- gam.check(m_binary2)

# Advanced interaction model - smooth interaction between depression and race
m_interaction2 <- bam(
  cbind(GOP_votes, DEM_votes) ~ s(depr_prop_std, white_prop_std, k=10) + s(STNAME, bs="re"),
  data = uv,
  family = binomial(link="logit"),
  discrete = TRUE
)

# Optional: Spatial smoother if coordinates are available
if(all(c("longitude", "latitude") %in% names(uv))) {
  m_spatial <- bam(
    cbind(GOP_votes, DEM_votes) ~ depr_prop_std + white_prop_std + 
      s(longitude, latitude, bs="gp", k=100) + s(STNAME, bs="re"),
    data = uv,
    family = binomial(link="logit"),
    discrete = TRUE
  )
}

# --- 3. Check concurvity for multiple smooths ---
concurvity_check <- concurvity(m_smooth_depr2, full=TRUE)

# --- 4. Check for overdispersion ---
# Calculate residual deviance / residual df ratio
overdispersion_check <- sum(residuals(m_base2, type="deviance")^2) / m_base2$df.residual
cat("Overdispersion check (should be close to 1):", overdispersion_check, "\n")

# If overdispersion detected, consider switching to quasibinomial
use_quasi <- FALSE
if(overdispersion_check > 1.5) {
  m_base2_quasi <- update(m_base2, family=quasibinomial())
  cat("Overdispersion detected, quasi-likelihood model fitted\n")
  use_quasi <- TRUE
}

# --- 5. Inspect models ---
summary_base <- summary(m_base2)
summary_fixed <- summary(m_fixed2)
summary_smooth <- summary(m_smooth_depr2)
summary_slope <- summary(m_rand_slope2)
summary_binary <- summary(m_binary2)
summary_interaction <- summary(m_interaction2)

# Compare models - only use AIC if not using quasi models
if(!use_quasi) {
  aic_comparison <- AIC(m_fixed2, m_base2, m_smooth_depr2, m_rand_slope2, m_interaction2)
  BIC_comparison <- BIC(m_fixed2, m_base2, m_smooth_depr2, m_rand_slope2, m_interaction2)
  
  # Identify best model based on AIC
  best_model_name <- names(which.min(aic_comparison))
  best_model <- get(best_model_name)
} else {
  # If using quasi models, cannot use AIC/BIC
  cat("Using quasi-likelihood models due to overdispersion. AIC/BIC not applicable.\n")
  # Choose smooth depression model as best for visualization
  best_model_name <- "m_smooth_depr2"
  best_model <- m_smooth_depr2
  
  # Create placeholders for model comparison
  aic_comparison <- "Not applicable with quasi-likelihood models"
  BIC_comparison <- "Not applicable with quasi-likelihood models"
}

# --- 6. Extract random slopes for state selection ---
# Get random slopes from the model
rand_slopes <- NULL
tryCatch({
  rand_slopes <- ranef(m_rand_slope2)
  # Process the random effects
  if(!is.null(rand_slopes)) {
    # Find the column with random slopes (should be the second one, for by=depr_prop_std)
    slope_col <- ifelse(ncol(rand_slopes) > 1, 2, 1)
    slope_data <- data.frame(
      state = rownames(rand_slopes),
      effect = rand_slopes[,slope_col],
      stringsAsFactors = FALSE
    )
    
    # Sort states by the strength of depression effect
    slope_data <- slope_data[order(slope_data$effect, decreasing=TRUE),]
    
    # Select states based on effect pattern
    num_states <- nrow(slope_data)
    meaningful_states <- c(
      slope_data$state[1:min(3, num_states)],  # Top 3 (or fewer) positive effects
      slope_data$state[floor(num_states/2) + c(-1,0,1)],  # Middle 3 states (if enough states)
      slope_data$state[max(1, num_states-2):num_states]  # Bottom 3 (or fewer) negative effects
    )
    # Remove duplicates
    meaningful_states <- unique(meaningful_states)
  } else {
    # Fallback to random selection if ranef() fails
    meaningful_states <- sample(levels(uv$STNAME), min(9, length(levels(uv$STNAME))))
  }
}, error = function(e) {
  cat("Error extracting random slopes:", e$message, "\n")
  # Fallback to random selection
  meaningful_states <- sample(levels(uv$STNAME), min(9, length(levels(uv$STNAME))))
})

# If random slopes couldn't be extracted, use a simple approach
if(is.null(rand_slopes)) {
  # Create a simple linear model for each state and extract slopes
  slope_data <- data.frame(state=character(), effect=numeric(), stringsAsFactors=FALSE)
  
  for(state in levels(uv$STNAME)) {
    subset_data <- subset(uv, STNAME == state)
    if(nrow(subset_data) > 5) {  # Only fit if enough data
      tryCatch({
        m_temp <- lm(per_gop_prop ~ depr_prop, data=subset_data, weights=total_votes)
        new_row <- data.frame(state=state, effect=coef(m_temp)["depr_prop"], stringsAsFactors=FALSE)
        slope_data <- rbind(slope_data, new_row)
      }, error=function(e) {
        # Skip if error
      })
    }
  }
  
  if(nrow(slope_data) > 0) {
    slope_data <- slope_data[order(slope_data$effect, decreasing=TRUE),]
    num_states <- nrow(slope_data)
    meaningful_states <- c(
      slope_data$state[1:min(3, num_states)],  # Top 3 (or fewer) positive effects
      slope_data$state[floor(num_states/2) + c(-1,0,1)],  # Middle 3 states (if enough states)
      slope_data$state[max(1, num_states-2):num_states]  # Bottom 3 (or fewer) negative effects
    )
    meaningful_states <- unique(meaningful_states)
  } else {
    # Last resort: just pick some states randomly
    meaningful_states <- sample(levels(uv$STNAME), min(9, length(levels(uv$STNAME))))
  }
}

cat("Selected states for slope visualization:", paste(meaningful_states, collapse=", "), "\n")

# --- 7. Generate plots for depression effect (using best model) ---
# Create sequence for prediction
newd <- data.frame(
  depr_prop_std = seq(min(uv$depr_prop_std), max(uv$depr_prop_std), length=100),
  white_prop_std = 0,  # Mean of standardized variable is 0
  STNAME = levels(uv$STNAME)[1] # Reference state
)

# Add original scale for plotting
depr_mean <- mean(uv$depr_prop)
depr_sd <- sd(uv$depr_prop)
newd$depr_prop <- newd$depr_prop_std * depr_sd + depr_mean

# Generate predictions from best model
pr <- predict(best_model, newdata=newd, type="link", se.fit=TRUE)
newd$fit <- plogis(pr$fit) 
newd$upper95 <- plogis(pr$fit + 1.96*pr$se.fit)
newd$lower95 <- plogis(pr$fit - 1.96*pr$se.fit)

# Extract effect size and p-value for annotation
if (best_model_name %in% c("m_base2", "m_fixed2")) {
  # For models with linear depression effect
  effect_size <- round(coef(best_model)["depr_prop_std"] * 100, 2) # Convert to percentage points
  p_value <- summary(best_model)$p.table["depr_prop_std", "Pr(>|z|)"]
  effect_annotation <- sprintf("Effect: %.2f pp per SD in depression (p = %.4f)", effect_size, p_value)
} else {
  # For models with smooth depression effect
  smooth_p <- summary(best_model)$s.table["s(depr_prop_std)", "p-value"]
  effect_annotation <- sprintf("Non-linear effect of depression (p = %.4f)", smooth_p)
}

# Convert back to percentages for plotting
newd$fit_pct <- newd$fit * 100
newd$upper95_pct <- newd$upper95 * 100
newd$lower95_pct <- newd$lower95 * 100
newd$depr_pct <- newd$depr_prop * 100

# Create improved ggplot version with annotation
p_effect <- ggplot(newd, aes(x=depr_pct, y=fit_pct)) +
  geom_ribbon(aes(ymin=lower95_pct, ymax=upper95_pct), alpha=0.2) +
  geom_line(size=1, color="blue") +
  annotate("text", x = min(newd$depr_pct) + 1, y = max(newd$fit_pct) - 5, 
           label = effect_annotation, hjust = 0, size = 3, fontface = "bold") +
  labs(
    title="Partial Effect of Depression Rate on Trump Vote %",
    subtitle=paste("Based on", best_model_name, "(controlling for race and state effects)"),
    x="County Depression Rate (%)",
    y="Predicted Trump Vote (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face="bold"),
    axis.title = element_text(face="bold")
  )

# --- 8. Create state-specific effects plot using selected states ---
# Generate predictions for these specific states
states_effects <- data.frame()

for (state in meaningful_states) {
  newd_state <- data.frame(
    depr_prop_std = seq(min(uv$depr_prop_std), max(uv$depr_prop_std), length=20),
    white_prop_std = 0,
    STNAME = state
  )
  
  # Add back original scale for plotting
  newd_state$depr_prop <- newd_state$depr_prop_std * depr_sd + depr_mean
  newd_state$depr_pct <- newd_state$depr_prop * 100
  
  pr_state <- predict(m_rand_slope2, newdata=newd_state, type="link", se.fit=TRUE)
  newd_state$fit <- plogis(pr_state$fit) * 100
  newd_state$state <- state
  
  states_effects <- rbind(states_effects, newd_state)
}

# Create state effects plot with more meaningful state selection
p_slopes <- ggplot(states_effects, aes(x=depr_pct, y=fit, color=state, group=state)) +
  geom_line(size=1) +
  labs(
    title="State-Specific Effects of Depression on Trump Vote %",
    subtitle="Selected states showing variation in depression effects",
    x="County Depression Rate (%)",
    y="Predicted Trump Vote (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face="bold"),
    axis.title = element_text(face="bold"),
    legend.title = element_blank(),
    legend.position = "right"
  )

# --- 9. Create advanced interaction effect plot ---
# Generate grid for interaction visualization
grid_size <- 30
interaction_grid <- expand.grid(
  depr_prop_std = seq(min(uv$depr_prop_std), max(uv$depr_prop_std), length=grid_size),
  white_prop_std = seq(min(uv$white_prop_std), max(uv$white_prop_std), length=grid_size),
  STNAME = levels(uv$STNAME)[1]
)

# Convert back to original scale for plotting
interaction_grid$depr_prop <- interaction_grid$depr_prop_std * depr_sd + depr_mean
interaction_grid$depr_pct <- interaction_grid$depr_prop * 100

white_mean <- mean(uv$white_prop)
white_sd <- sd(uv$white_prop)
interaction_grid$white_prop <- interaction_grid$white_prop_std * white_sd + white_mean
interaction_grid$white_pct <- interaction_grid$white_prop * 100

# Get predictions
interaction_pred <- predict(m_interaction2, newdata=interaction_grid, type="response")
interaction_grid$fit <- interaction_pred * 100

# Create heatmap for smooth interaction
p_interaction <- ggplot(interaction_grid, aes(x=depr_pct, y=white_pct, fill=fit)) +
  geom_tile() +
  scale_fill_viridis(name="Predicted\nTrump Vote %") +
  labs(
    title="Smooth Interaction Between Depression and Race on Trump Vote %",
    x="County Depression Rate (%)",
    y="White Population (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face="bold"),
    axis.title = element_text(face="bold")
  )

# --- 10. Create weighted binary outcome plot ---
# Generate predictions for binary model
newd_binary <- data.frame(
  depr_prop_std = seq(min(uv$depr_prop_std), max(uv$depr_prop_std), length=100),
  white_prop_std = 0,
  STNAME = levels(uv$STNAME)[1]
)

# Add back original scale for plotting
newd_binary$depr_prop <- newd_binary$depr_prop_std * depr_sd + depr_mean
newd_binary$depr_pct <- newd_binary$depr_prop * 100

pr_binary <- predict(m_binary2, newdata=newd_binary, type="link", se.fit=TRUE)
newd_binary$fit <- plogis(pr_binary$fit)
newd_binary$upper95 <- plogis(pr_binary$fit + 1.96*pr_binary$se.fit)
newd_binary$lower95 <- plogis(pr_binary$fit - 1.96*pr_binary$se.fit)

p_binary <- ggplot(newd_binary, aes(x=depr_pct, y=fit)) +
  geom_ribbon(aes(ymin=lower95, ymax=upper95), alpha=0.2) +
  geom_line(size=1, color="red") +
  labs(
    title="Probability of Republican Majority by Depression Rate",
    subtitle="Weighted by county total votes",
    x="County Depression Rate (%)",
    y="Probability of GOP Majority"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face="bold"),
    axis.title = element_text(face="bold")
  )

# --- 11. Create central figure (improved visualization) ---
# Create a scatterplot of raw data
p_scatter <- ggplot(uv, aes(x=depr_prop*100, y=per_gop_prop*100, size=total_votes)) +
  geom_point(alpha=0.5, aes(color=STNAME)) +
  geom_smooth(method="lm", se=TRUE, color="black", size=1) +
  scale_size_continuous(range=c(0.5, 3), name="County Size\n(Total Votes)") +
  labs(
    title="Relationship Between Depression and Trump Vote Share",
    subtitle=paste("Data from", nrow(uv), "counties across", length(levels(uv$STNAME)), "states"),
    x="Depression Rate (%)",
    y="Trump Vote (%)",
    color="State"
  ) +
  theme_minimal() +
  theme(legend.position="right")

# Combine with effect plots for central figure (improved for report)
central_figure <- tryCatch({
  gridExtra::grid.arrange(
    p_scatter,
    p_effect,
    p_slopes,
    p_interaction,
    ncol=2,
    layout_matrix = rbind(c(1,1), c(2,3), c(4,4)),
    widths = c(1, 1),
    heights = c(1, 1, 1)
  )
}, error = function(e) {
  cat("Error creating central figure:", e$message, "\n")
  # Fallback to simpler layout
  gridExtra::grid.arrange(p_scatter, p_effect, p_slopes, p_interaction, ncol=2)
})

# Extended central figure with all plots (for supplementary materials)
extended_figure <- tryCatch({
  gridExtra::grid.arrange(
    p_scatter,
    p_effect,
    p_slopes,
    p_interaction,
    p_binary,
    ncol=3,
    layout_matrix = rbind(c(1,1,1), c(2,3,4), c(5,5,5)),
    widths = c(1, 1, 1),
    heights = c(1, 1, 1)
  )
}, error = function(e) {
  cat("Error creating extended figure:", e$message, "\n")
  # Fallback to simpler layout
  gridExtra::grid.arrange(p_scatter, p_effect, p_slopes, p_interaction, p_binary, ncol=2)
})

# --- 12. Residual diagnostics ---
# Extract residuals from best model
uv$residuals <- residuals(best_model, type="deviance")
uv$fitted <- fitted(best_model)

# Create residual diagnostic plots
p1 <- ggplot(uv, aes(x=fitted, y=residuals)) +
  geom_point(alpha=0.5) +
  geom_hline(yintercept=0, linetype="dashed", color="red") +
  geom_smooth(se=FALSE, color="blue", method="loess") +
  labs(title="Residuals vs Fitted", 
       subtitle=paste("Based on", best_model_name),
       x="Fitted values", y="Deviance residuals") +
  theme_minimal()

p2 <- ggplot(uv, aes(sample=residuals)) +
  stat_qq() +
  stat_qq_line(color="red") +
  labs(title="Normal Q-Q Plot", 
       subtitle="Checking normality of residuals",
       x="Theoretical Quantiles", y="Deviance residuals") +
  theme_minimal()

p3 <- ggplot(uv, aes(x=residuals)) +
  geom_histogram(bins=30, fill="lightblue", color="black") +
  labs(title="Histogram of Residuals", 
       x="Deviance residuals", y="Count") +
  theme_minimal()

# Combine residual plots
residual_grid <- grid.arrange(p1, p2, p3, ncol=2)

# --- 13. Save outputs for R Markdown ---
# Save all key objects for the R Markdown document
save(uv, m_base2, m_fixed2, m_smooth_depr2, m_rand_slope2, m_binary2, m_interaction2,
     p_effect, p_slopes, p_scatter, p_interaction, p_binary,
     central_figure, extended_figure, residual_grid, concurvity_check,
     gam_check_smooth, gam_check_binary, overdispersion_check,
     summary_base, summary_fixed, summary_smooth, summary_slope, 
     summary_binary, summary_interaction, 
     aic_comparison, BIC_comparison, best_model_name,
     slope_data, meaningful_states,  # Save the state selection info for reporting
     file="analysis_objects.RData")

# Save key plots for inclusion in the report
ggsave("central_figure.png", central_figure, width=12, height=10)
ggsave("extended_figure.png", extended_figure, width=14, height=12)
ggsave("depression_effect.png", p_effect, width=8, height=6)
ggsave("state_effects.png", p_slopes, width=8, height=6)
ggsave("interaction_effect.png", p_interaction, width=8, height=6)
ggsave("binary_outcome.png", p_binary, width=8, height=6)
ggsave("residual_diagnostics.png", residual_grid, width=10, height=8)

cat("Analysis complete. Objects saved for R Markdown report.\n")
cat("Best model according to AIC:", best_model_name, "\n")
cat("State selection for random slopes plot: ", paste(meaningful_states, collapse=", "), "\n")
cat("Overdispersion check ratio: ", overdispersion_check, "\n")
