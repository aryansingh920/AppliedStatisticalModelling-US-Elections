#!/usr/bin/env Rscript

# --- Set CRAN mirror & install required packages if missing ---
options(repos="https://cloud.r-project.org")
required_packages <- c("mgcv", "readr", "ggplot2", "sf", "viridis", "gridExtra")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly=TRUE)) install.packages(pkg)
}

library(mgcv)
library(readr)
library(ggplot2)
library(sf)
library(viridis)
library(gridExtra)

# --- 1. Read and prepare data ---
uv <- read_csv("US Votes Data.csv", show_col_types=FALSE)

# Make proportions
uv$per_gop_prop <- uv$per_gop / 100
uv$depr_prop <- uv$Crude.Prevalence.Estimate / 100
uv$white_prop <- uv$race

# Ensure states are a factor
uv$STNAME <- as.factor(uv$STNAME)

# --- 2. Fit the model with a state random intercept via bs="re" ---
# Quasi-binomial on logit, weighted by county turnout
m_bam <- bam(
  per_gop_prop ~ depr_prop + white_prop + s(STNAME, bs="re"),
  data = uv,
  family = quasibinomial(link="logit"),
  weights = total_votes,
  discrete=TRUE # speeds up fitting on large data
)

# --- 3. Inspect ---
summary_model <- summary(m_bam)
print(summary_model)
pdf("model_terms.pdf")
plot(m_bam, pages=1, shade=TRUE)
dev.off()

# --- 4. Generate enhanced partial effect plot for depression ---
newd <- data.frame(
  depr_prop = seq(min(uv$depr_prop), max(uv$depr_prop), length=100),
  white_prop = mean(uv$white_prop),
  STNAME = levels(uv$STNAME)[1] # any one state for intercept
)

pr <- predict(m_bam, newdata=newd, type="link", se.fit=TRUE)
newd$fit <- plogis(pr$fit) 
newd$upper95 <- plogis(pr$fit + 1.96*pr$se.fit)
newd$lower95 <- plogis(pr$fit - 1.96*pr$se.fit)

# Convert back to percentages for plotting
newd$fit_pct <- newd$fit * 100
newd$upper95_pct <- newd$upper95 * 100
newd$lower95_pct <- newd$lower95 * 100
newd$depr_pct <- newd$depr_prop * 100

# Create improved ggplot version
p_effect <- ggplot(newd, aes(x=depr_pct, y=fit_pct)) +
  geom_ribbon(aes(ymin=lower95_pct, ymax=upper95_pct), alpha=0.2) +
  geom_line(size=1, color="blue") +
  labs(
    title="Partial Effect of Depression Rate on Trump Vote %",
    subtitle="Controlling for race and state random effects",
    x="County Depression Rate (%)",
    y="Predicted Trump Vote (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face="bold"),
    axis.title = element_text(face="bold")
  )

# Save both base R and ggplot versions
png("depression_effect.png", width=800, height=600)
plot(newd$depr_prop, newd$fit, type="l", xlab="County Depression Rate", 
     ylab="Predicted Trump Vote %", main="Partial Effect of Depression")
lines(newd$depr_prop, newd$upper95, lty=2)
lines(newd$depr_prop, newd$lower95, lty=2)
dev.off()

ggsave("depression_effect_ggplot.png", p_effect, width=8, height=6)

# --- 5. Residual diagnostics ---
# Extract residuals
uv$residuals <- residuals(m_bam, type="deviance")
uv$fitted <- fitted(m_bam)

# Create basic residual diagnostic plots
p1 <- ggplot(uv, aes(x=fitted, y=residuals)) +
  geom_point(alpha=0.5) +
  geom_hline(yintercept=0, linetype="dashed", color="red") +
  geom_smooth(se=FALSE, color="blue", method="loess") +
  labs(title="Residuals vs Fitted", x="Fitted values", y="Deviance residuals") +
  theme_minimal()

p2 <- ggplot(uv, aes(sample=residuals)) +
  stat_qq() +
  stat_qq_line(color="red") +
  labs(title="Normal Q-Q Plot", x="Theoretical Quantiles", y="Deviance residuals") +
  theme_minimal()

p3 <- ggplot(uv, aes(x=residuals)) +
  geom_histogram(bins=30, fill="lightblue", color="black") +
  labs(title="Histogram of Residuals", x="Deviance residuals", y="Count") +
  theme_minimal()

# Combine and save residual plots
residual_grid <- grid.arrange(p1, p2, p3, ncol=2)
ggsave("residual_diagnostics.png", residual_grid, width=10, height=8)

# --- 6. Spatial visualization of residuals ---
# Check if spatial data is available in the dataset
if ("LONGITUDE" %in% names(uv) && "LATITUDE" %in% names(uv)) {
  # Create spatial residual plot
  p_spatial <- ggplot(uv, aes(x=LONGITUDE, y=LATITUDE, color=residuals)) +
    geom_point(alpha=0.7) +
    scale_color_viridis() +
    labs(title="Spatial Distribution of Residuals", 
         x="Longitude", y="Latitude", color="Residual") +
    theme_minimal()
  
  ggsave("spatial_residuals.png", p_spatial, width=8, height=6)
} else {
  cat("Spatial coordinates not found in dataset. Skipping spatial residual plot.\n")
}

# --- 7. Optional: Refit with spatial smoother ---
# Try to incorporate spatial information if available
if ("LONGITUDE" %in% names(uv) && "LATITUDE" %in% names(uv)) {
  # Model with spatial smoother
  m_spatial <- bam(
    per_gop_prop ~ depr_prop + white_prop + 
      s(STNAME, bs="re") + 
      s(LONGITUDE, LATITUDE, bs="tp", k=30),
    data = uv,
    family = quasibinomial(link="logit"),
    weights = total_votes,
    discrete=TRUE
  )
  
  # Compare models
  cat("\n\nModel comparison (with vs. without spatial smoother):\n")
  print(AIC(m_bam, m_spatial))
  
  # Check if spatial effect is significant
  summary_spatial <- summary(m_spatial)
  print(summary_spatial)
  
  # Plot spatial effect
  pdf("spatial_effect.pdf")
  plot(m_spatial, select=3)  # The third smooth term should be the spatial effect
  dev.off()
  
  # Re-extract depression effect for comparison
  newd_spatial <- data.frame(
    depr_prop = seq(min(uv$depr_prop), max(uv$depr_prop), length=100),
    white_prop = mean(uv$white_prop),
    STNAME = levels(uv$STNAME)[1],
    LONGITUDE = mean(uv$LONGITUDE),
    LATITUDE = mean(uv$LATITUDE)
  )
  
  pr_spatial <- predict(m_spatial, newdata=newd_spatial, type="link", se.fit=TRUE)
  newd_spatial$fit <- plogis(pr_spatial$fit) * 100
  newd_spatial$upper95 <- plogis(pr_spatial$fit + 1.96*pr_spatial$se.fit) * 100
  newd_spatial$lower95 <- plogis(pr_spatial$fit - 1.96*pr_spatial$se.fit) * 100
  newd_spatial$depr_pct <- newd_spatial$depr_prop * 100
  
  # Plot comparison
  spatial_comparison <- data.frame(
    depr_pct = c(newd$depr_pct, newd_spatial$depr_pct),
    fit = c(newd$fit_pct, newd_spatial$fit),
    upper95 = c(newd$upper95_pct, newd_spatial$upper95),
    lower95 = c(newd$lower95_pct, newd_spatial$lower95),
    model = rep(c("Base Model", "Spatial Model"), each=100)
  )
  
  p_compare <- ggplot(spatial_comparison, aes(x=depr_pct, y=fit, color=model, fill=model)) +
    geom_ribbon(aes(ymin=lower95, ymax=upper95), alpha=0.2) +
    geom_line(size=1) +
    labs(
      title="Effect of Depression Rate on Trump Vote %: Model Comparison",
      subtitle="Comparing models with and without spatial smoother",
      x="County Depression Rate (%)",
      y="Predicted Trump Vote (%)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face="bold"),
      axis.title = element_text(face="bold"),
      legend.title = element_blank(),
      legend.position = "bottom"
    )
  
  ggsave("depression_effect_comparison.png", p_compare, width=8, height=6)
}

# --- 8. Optional: Model with random slopes by state ---
# Fit model with random slopes for depression by state
m_rand_slope <- bam(
  per_gop_prop ~ white_prop + 
    s(STNAME, bs="re") +                      # Random intercept
    s(STNAME, depr_prop, bs="re"),            # Random slope
  data = uv,
  family = quasibinomial(link="logit"),
  weights = total_votes,
  discrete=TRUE
)

# Examine summary
summary_rs <- summary(m_rand_slope)
print(summary_rs)

# Plot random effects
pdf("random_slopes.pdf")
plot(m_rand_slope, pages=1)
dev.off()

# Extract state-specific depression effects
states <- levels(uv$STNAME)
state_effects <- data.frame()

for (state in states) {
  newd_state <- data.frame(
    depr_prop = seq(min(uv$depr_prop), max(uv$depr_prop), length=20),
    white_prop = mean(uv$white_prop),
    STNAME = state
  )
  
  pr_state <- predict(m_rand_slope, newdata=newd_state, type="link", se.fit=TRUE)
  newd_state$fit <- plogis(pr_state$fit) * 100
  newd_state$state <- state
  newd_state$depr_pct <- newd_state$depr_prop * 100
  
  state_effects <- rbind(state_effects, newd_state)
}

# Plot state-specific slopes
# Limit to 10 random states for clarity if there are many
if (length(states) > 10) {
  set.seed(123)
  plot_states <- sample(states, 10)
  state_effects_plot <- subset(state_effects, state %in% plot_states)
} else {
  state_effects_plot <- state_effects
}

p_slopes <- ggplot(state_effects_plot, aes(x=depr_pct, y=fit, color=state, group=state)) +
  geom_line() +
  labs(
    title="State-Specific Effects of Depression on Trump Vote %",
    subtitle="Random slopes model by state",
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

ggsave("state_specific_effects.png", p_slopes, width=10, height=8)

# --- 9. Save consolidated results ---
# Create a summary of key findings
sink("model_comparison_summary.txt")
cat("MODEL COMPARISON SUMMARY\n")
cat("=======================\n\n")

cat("1. BASE MODEL WITH STATE RANDOM INTERCEPTS\n")
cat("------------------------------------------\n")
print(summary(m_bam))
cat("\n\n")

if (exists("m_spatial")) {
  cat("2. MODEL WITH SPATIAL SMOOTHER\n")
  cat("------------------------------\n")
  print(summary(m_spatial))
  cat("\n\n")
}

cat("3. MODEL WITH RANDOM SLOPES BY STATE\n")
cat("-----------------------------------\n")
print(summary(m_rand_slope))
cat("\n\n")

# Calculate model fit metrics for comparison
if (exists("m_spatial")) {
  cat("MODEL FIT COMPARISON:\n")
  cat("--------------------\n")
  
  # Extract deviance and df for each model
  models <- list(Base=m_bam, Spatial=m_spatial, RandomSlopes=m_rand_slope)
  comparison <- data.frame(
    Model = names(models),
    Deviance = sapply(models, function(m) sum(residuals(m, type="deviance")^2)),
    DF = sapply(models, function(m) summary(m)$edf),
    AIC = sapply(models, AIC)
  )
  
  print(comparison)
}
sink()

cat("\nAnalysis complete. Check output files for results.\n")
