<h1 align="center">Modeling Wine Quality Through Chemical Composition: A Dual Approach</h1>




<div align="center">

<table>
  <tr>
    <td colspan="2" align="center" style="background-color: white; color: black;"><strong>Table of Contents</strong></td>
  </tr>
  <tr>
    <td style="background-color: white; color: black; padding: 10px;">1. <a href="#project-objective" style="color: black;">Overview</a></td>
    <td style="background-color: gray; color: black; padding: 10px;">6. <a href="#xgboost" style="color: black;">XGBoost</a></td>
  </tr>
  <tr>
    <td style="background-color: gray; color: black; padding: 10px;">2. <a href="#data-description-and-preprocessing" style="color: black;">Data Description and Preprocessing</a></td>
    <td style="background-color: white; color: black; padding: 10px;">7. <a href="#comparative-summary" style="color: black;">Comparative Summary</a></td>
  </tr>
  <tr>
    <td style="background-color: white; color: black; padding: 10px;">3. <a href="#logistic-regression" style="color: black;">Logistic Regression</a></td>
    <td style="background-color: gray; color: black; padding: 10px;">8. <a href="#conclusion" style="color: black;">Conclusion</a></td>
  </tr>
  <tr>
    <td style="background-color: gray; color: black; padding: 10px;">4. <a href="#interpretation-and-results" style="color: black;">Interpretation and Results</a></td>
    <td style="background-color: white; color: black; padding: 10px;">9. <a href="#diagnostics-and-model-validity" style="color: black;">Diagnostics and Model Validity</a></td>
  </tr>
  <tr>
    <td style="background-color: white; color: black; padding: 10px;">5. <a href="#final-model-specification" style="color: black;">Final Model Specification</a></td>
    <td style="background-color: gray; color: black; padding: 10px;">10. <a href="#future-work" style="color: black;">Future Work</a></td>
  </tr>
</table>

</div>




## Project Objective

This project investigates which **chemical properties** make a wine **high quality**—defined here as wines with a quality rating of **7 or higher**. Instead of relying on a single modeling strategy, we approach this problem from **two complementary perspectives**:

- **Logistic Regression (GLM)** emphasizes **statistical interpretability**, marginal effects, and hypothesis testing.
- **XGBoost** captures **nonlinear relationships** and potential **feature interactions**, offering a flexible machine learning approach.

Each model targets a different angle of the same core question: *What measurable characteristics separate great wines from average ones?*

By comparing these approaches, we aim to **extract a stable, converging set of chemical features** that consistently predict wine quality—while maintaining model validity and minimizing overfitting.



## Data Description and Preprocessing

- No missing values
- Non-normal distributions across predictors
- High multicollinearity, especially among acidity, alcohol, and density
- Binary target variable defined as:

```math
Y_i =
\begin{cases}
1 & \text{if quality}_i \geq 7 \\
0 & \text{otherwise}
\end{cases}
```

### Class Distribution

- Approximately 13% of wines are classified as high quality



## Logistic Regression

### Initial Specification

Let $\( \mathbf{X}_i \)$ represent the chemical features of wine $\( i \)$. The model is:

$$
\log\left(\frac{\Pr(Y_i = 1)}{\Pr(Y_i = 0)}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j X_{ij}
$$

Initial modeling revealed:
- High **multicollinearity** (VIFs > 10)
- Low **recall** (~37%) for high-quality wines
- Overfitting risk from redundant features




### 1. Reducing Multicollinearity with Feature Combination


A composite feature was constructed:

```math
\text{density\_combined}_i = 0.6815 \cdot \text{FixedAcidity}_i - 0.4947 \cdot \text{Alcohol}_i
```

**Why this helps**:
- Reduces the dimension of the collinear subspace  
- Captures shared variance in a single predictor  
- Improves numerical stability of coefficient estimation  

Rather than dropping one variable arbitrarily, this preserves useful variation while avoiding redundancy.



### 2. Stepwise Feature Selection

Stepwise AIC (both forward and backward) yielded the final predictors:
- Volatile Acidity
- Citric Acid
- Chlorides
- Total Sulfur Dioxide
- pH
- Sulphates
- Alcohol

**Why this helps**:
- Excludes variables that do not contribute meaningful independent signal  
- Automatically removes predictors whose contribution is confounded by others  
- Reduces the chance of including variables that exacerbate multicollinearity





### 3. Validation: Retaining Predictive Power (ROC Curve)

**Why this matters**:  
When reducing multicollinearity—whether by combining features or removing redundant variables—there is a risk of **omitting important signal** along with the noise. This can degrade the model’s ability to distinguish between classes.

By validating against the **Receiver Operating Characteristic (ROC) curve**, we ensured that:
- The simplified model **retained its discriminative ability**
- There was **no loss in true positive performance**
- Tradeoffs made for interpretability and statistical soundness **did not compromise prediction quality**

This step confirmed that our model improvements enhanced **robustness and stability** without sacrificing predictive effectiveness.



### 4. Addressing Class Imbalance with SMOTE

- SMOTE applied with $\( k = 5 \)$ neighbors
- Balanced the minority class during training


**Why this matters**:  
The dataset is highly imbalanced—only ~13% of wines are labeled high quality. Without balancing, the model would be biased toward predicting the majority class (low-quality wines), resulting in **low recall** for the positive class.  

SMOTE (Synthetic Minority Oversampling Technique) generates new synthetic examples of the minority class, allowing the model to:
- Learn better **decision boundaries**
- Reduce **false negatives**
- Treat high-quality wines with **equal attention** during training



### 5. Threshold Optimization for Recall

We optimized the classification threshold to maximize **recall**—our primary evaluation metric—rather than using the default threshold of 0.5. In an imbalanced classification setting, where only ~13% of wines are high quality, a 0.5 threshold leads to:
- High accuracy (due to correctly predicting the majority class)
- But **very low recall** for the minority class (many false negatives)

**Optimization Process**:

- We evaluated model performance over a **range of thresholds** from 0.1 to 0.5
- For each threshold, we computed:
  - Recall (true positive rate)
  - Precision
  - Accuracy

  
- The best **tradeoff between recall and acceptable precision** was achieved at a threshold of **0.2** 
  - This aligns with the real-world objective: it’s worse to **miss a great wine** than to mistakenly overrate a mediocre one



### 6. Post-SMOTE ROC Validation

- ROC AUC remained consistent
- Cross-validation confirmed no overfitting from synthetic resampling

**Why this matters**:  
Although SMOTE balances the classes, it introduces artificial data. If not validated carefully, this can lead to **overfitting** or **inflated performance metrics**.  

By re-checking the ROC AUC and applying **cross-validation**:
- We confirmed that the model generalized well  
- There was **no leakage or distortion** from synthetic samples  
- Performance improvements were genuine and not artifacts of oversampling



### Final Model Specification

Let:

```math
\mathbf{X}_i = [\text{VolatileAcidity},\, \text{CitricAcid},\, \text{Chlorides},\, \text{TotalSulfurDioxide},\, \text{pH},\, \text{Sulphates},\, \text{Alcohol}]
```

Then:

```math
\log\left(\frac{\Pr(Y_i = 1)}{\Pr(Y_i = 0)}\right) = \beta_0 + \beta_1 \cdot \text{VolatileAcidity}_i + \ldots + \beta_7 \cdot \text{Alcohol}_i
```


Together, these steps ensured that:
- All retained features had **Variance Inflation Factors (VIFs) < 2.5**  
- The final model was **stable, interpretable, and statistically valid**  
- Coefficient estimates were more reliable and easier to interpret in terms of real-world chemical effects




### Interpretation and Results

#### 1. Coefficient Interpretation

| Predictor             | Estimate (β) | Odds Ratio | Direction | Interpretation                                     |
|-----------------------|--------------|------------|-----------|---------------------------------------------------|
| Alcohol               | +1.07        | 2.91       | ↑         | 1% increase in alcohol → nearly 3× odds           |
| Sulphates             | +3.28        | 26.6       | ↑         | 1-unit increase → 26.6× odds                      |
| Citric Acid           | +1.36        | 3.90       | ↑         | Moderate positive impact                          |
| Chlorides             | −2.52        | 0.08       | ↓         | 0.01-unit increase → ~12.4% decrease in odds      |
| Volatile Acidity      | −3.25        | 0.04       | ↓         | Strong negative effect                            |
| pH                    | −1.59        | 0.20       | ↓         | Higher pH associated with lower odds              |
| Total Sulfur Dioxide  | −0.01        | ~0.99      | ↓         | Slight negative effect                            |


All predictors are statistically significant at **p < 0.05**.



#### 2. Model Fit: McFadden’s $\( R^2 \)$

McFadden’s pseudo- $\( R^2 \)$ is defined as:

```math
R^2_{\text{McFadden}} = 1 - \frac{\log L_{\text{model}}}{\log L_{\text{null}}}
```

- Estimated $\( R^2_{\text{McFadden}} \approx 0.368 \)$
- Indicates strong model fit in the context of classification models



### Final Model Note


We explored including **interaction terms** (e.g., `FixedAcidity × Alcohol`, `CitricAcid × SulfurDioxide`), but these:
- **Increased VIFs by an order of magnitude**
- Led to **unstable standard errors**
- Did **not meaningfully reduce AIC**

As a result, we continued with a **main-effects model** that balances interpretability and statistical soundness. Significant interaction terms after rerunning the stepwise selection were indentified purely for contextual insights.


## XGBoost

### Initial Model Specification

XGBoost minimizes the following regularized objective:

```math
\mathcal{L}(\phi) = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i) + \sum_{t=1}^{T} \Omega(f_t)
```

Where:

- $\( \ell \)$ is the logistic loss
- $\( f_t \)$ is the prediction function of tree $\( t \)$
- $\( \Omega(f_t)$ = $\gamma T + \frac{1}{2} \lambda \sum_j w_j^2 \)$ is the regularization penalty



### 1. Initial Model Performance

- High accuracy  
- **Low recall for high-quality wines**  
- Threshold = 0.5 was not suitable due to class imbalance

**Why this matters**:  
Out-of-the-box, XGBoost maximized overall accuracy—but this came at the expense of identifying high-quality wines. The imbalance led to a strong bias toward the majority class (Class 0), making the model unreliable for our true objective. This motivated further adjustments to **rebalance the signal and prioritize recall**.



### 2. Addressing Class Imbalance with SMOTE

- SMOTE applied to training set  
- Better representation of Class 1 allowed the model to learn its characteristics

**Why this matters**:  
Like in the logistic model, oversampling the minority class using SMOTE improved the model’s ability to:
- Recognize patterns specific to high-quality wines  
- Avoid overfitting to synthetic instances by **only augmenting training data**  
- Improve **recall without distorting test set integrity**  

SMOTE helped the model generalize across quality tiers more effectively.



### 3. Threshold Optimization

- Threshold lowered to **0.2**
- Increased sensitivity to top-quality wines

**Why this matters**:  
Lowering the threshold shifts the model’s decision rule to favor more **true positives**. In our application, **false negatives are costly**—failing to detect a great wine is a more severe error than a false positive. Adjusting the threshold ensured the model aligned with this domain-specific priority.



### 4. Post-Tuning Performance Assessment

- ROC and Precision-Recall curves confirmed higher recall  
- Cross-validation confirmed model generalizability and robustness

**Why this matters**:  
After SMOTE and threshold tuning, it was critical to verify that performance gains were not artifacts of overfitting or class rebalancing. By:
- Plotting ROC and precision-recall curves  
- Evaluating performance via **10-fold cross-validation**

we ensured that improvements in recall were **genuine**, and the model remained **stable across different training splits**. This validation step confirmed that XGBoost was now **well-calibrated** for identifying high-quality wines in an imbalanced setting.



### 5. Model Interpretation with SHAP

SHAP (Shapley Additive exPlanations) is a game-theoretic approach to interpreting the output of complex machine learning models. It decomposes each model prediction into **contributions from each feature**, helping us understand **why** the model made a specific prediction.

Mathematically, SHAP expresses each prediction as:

```math
f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j x_j
```

Where:
- $\( \phi_0 \)$ is the model’s average output (baseline)
- $\( \phi_j \)$ is the **SHAP value** for feature $\( j \)$: how much that feature contributed (positively or negatively) to a specific prediction

**Why SHAP is important**:

- Unlike simpler metrics like **gain** (which tells you how often a feature is used in a tree split), SHAP:
  - Tells you **how much** and in **what direction** each feature influences each individual prediction
  - Is **additive and local**, allowing interpretation at both the sample level and model-wide
  - Can distinguish between **positive and negative contributions** for the same feature across different observations

In contrast, **gain** only captures how often a feature appears in high-information splits across trees—it doesn’t tell us whether that feature increases or decreases predictions or how it interacts with others.


### SHAP Results

| Rank | Feature              | SHAP Interpretation                    |
|------|----------------------|----------------------------------------|
| 1    | Alcohol              | Strong, monotonic positive effect      |
| 2    | Sulphates            | Strong, consistent positive effect     |
| 3    | Volatile Acidity     | Strong negative effect                 |
| 4    | Citric Acid          | Moderate positive impact               |
| 5    | Total Sulfur Dioxide| Small negative impact                   |




## Comparative Summary

| Rank | GLM Top Features     | XGBoost Top Features  |
|------|----------------------|-----------------------|
| 1    | Chlorides            | Alcohol               |
| 2    | Sulphates            | Sulphates             |
| 3    | Citric Acid          | Volatile Acidity      |
| 4    | Alcohol              | Citric Acid           |
| 5    | Volatile Acidity     | Sulfur Dioxide        |


**Key takeaways**:

- **Alcohol and Sulphates** appear as top contributors in both models and are consistent with wine chemistry theory—confirming their robustness.
- **Volatile Acidity** is also negatively associated with high quality across both approaches.
- **Citric Acid** and **Sulfur Dioxide** appear in both models but rank lower in influence.
- **Chlorides** appears important in the GLM due to its scale and statistical significance, but **SHAP revealed minimal actual contribution** in the tree-based model—highlighting the value of using multiple modeling lenses.
- **No interaction terms** emerged as meaningful in XGBoost, and all interaction terms explored in the GLM (e.g., Citric Acid × Sulfur Dioxide, Residual Sugar × pH) were dropped due to multicollinearity and minimal improvement in fit.
- Importantly, **only one observation was misclassified as a false negative in both models**, showing that the GLM and XGBoost agreed on nearly all borderline cases despite methodological differences.
- While GLM provides **statistical clarity and coefficient-level insight**, XGBoost allows for **flexibility and robustness to feature correlation**—offering a complementary perspective.

## Conclusion

This project used both **logistic regression** and **XGBoost** to identify which chemical attributes most reliably predict high-quality wines. Despite their differences, both models converged on the same core findings: **Alcohol**, **Sulphates**, and **Volatile Acidity** are the most impactful features, while **Citric Acid** and **Sulfur Dioxide** contribute moderately.

Through:
- Multicollinearity resolution via **feature combination**
- Model simplification via **stepwise selection**
- Rebalancing using **SMOTE**
- Threshold tuning to optimize **recall**
- Post-modeling validation using **ROC, SHAP, and cross-validation**

we ensured that both models were interpretable, stable, and generalizable.

Each model offered different strengths:
- **GLM** allowed us to interpret odds ratios and assess significance directly.
- **XGBoost**, enhanced by **SHAP**, revealed consistent patterns without needing explicit assumption checks.

Ultimately, this dual-model strategy allowed us to cross-validate conclusions, test robustness, and **build confidence in our understanding of which chemical features drive wine quality**. Both models aligned strongly despite structural differences—indicating that the insights extracted are not just statistically significant, but **chemically meaningful**.

## Diagnostics and Model Validity

- **GLM**:
  - All VIFs < 2.5
  - Residual diagnostics confirmed no misspecification
  - McFadden’s $\( R^2 \)$ ≈ 0.368 indicated a solid fit

- **XGBoost**:
  - SMOTE and threshold tuning improved recall
  - SHAP values confirmed feature reliability
  - No overfitting observed in cross-validation

- Only **one observation** was misclassified as a false negative by both models

## Future Work

- Apply **Bayesian GLMs** to estimate uncertainty and encode prior knowledge  
- Use **cost-sensitive learning** to penalize false negatives more heavily  
- Develop **ensemble models** combining GLM and XGBoost  
- Bootstrap SHAP values to assess **stability of feature importance**
