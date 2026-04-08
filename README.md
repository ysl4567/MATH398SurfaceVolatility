# FX Implied Volatility Surface Model

This project implements a **stochastic volatility-based option pricing framework** and constructs **implied volatility surfaces** for FX currency pairs.

The model combines:
- Perturbation-based stochastic volatility expansion
- Black–Scholes pricing
- Numerical implied volatility inversion
- Surface visualization and export tools

---

## Overview

The script `MATH398UpdatedV3.py` provides a full pipeline:

- **Model-based option pricing**
- **Implied volatility computation**
- **Volatility surface construction**
- **3D visualization**
- **CSV export of results**

The implementation is based on:

Stephen Diehl  
Volatility Surface Modeling  
https://www.stephendiehl.com/posts/volatility_surface/

---

## Model Components

### 1. Stochastic Volatility Backbone

Implements the deterministic variance process:

v₀(t) = θ + (V₀ − θ)e^{-κt}

This represents the mean-reverting variance structure.

---

### 2. Omega Operators

The model uses hierarchical integral operators:

- `omega_1` — single-layer integral  
- `omega_2` — double-layer integral  
- `omega_3` — triple-layer integral  

These are used to compute perturbation coefficients in the expansion.

---

### 3. Coefficient Construction

The following coefficients are computed:

- ψ (effective variance)
- a₀, a₁, a₂
- b₀, b₂

These coefficients feed into the pricing expansion.

---

### 4. Option Pricing Expansion

Implements a **perturbation expansion** around Black–Scholes:

- Base price: Black–Scholes put
- Higher-order corrections via numerical derivatives

Output:

P_model(K, T)

---

### 5. Black–Scholes Pricing

Standard European put option formula:

P = K e^{-rT} Φ(-d₂) − S₀ Φ(-d₁)

Used both for:
- Base pricing
- Implied volatility inversion

---

### 6. Implied Volatility Solver

Implied volatility is computed by solving:

BS_price(σ) = Model_price

Using:
- Brent’s method (`scipy.optimize.brentq`)

Features:
- Robust root-finding
- Adaptive volatility bounds

---

### 7. Volatility Surface Construction

Builds a grid:

- Strikes:  
  K ∈ [0.7S₀, 1.3S₀]

- Maturities:  
  T ∈ [0.05, 1.0]

For each (K, T):
1. Compute model price
2. Invert to implied volatility
3. Store in surface matrix

---

### 8. Visualization

Two types of surfaces:

#### Log-Moneyness Surface
- x-axis: ln(K / S₀)
- y-axis: maturity T
- z-axis: implied volatility

#### Absolute Strike Surface
- x-axis: strike K
- y-axis: maturity T
- z-axis: implied volatility

---

### 9. Data Export

Surfaces are saved as CSV files:

- Relative surface:
<Pair>_surface_values.csv

- Absolute surface:
<Pair>_absolute_surface_values.csv

Each row contains:
- Currency pair
- Maturity
- Strike
- Implied volatility

---

## FX Data Included

The model includes calibrated parameters for:

- USD/JPY
- USD/AUD

Each dataset contains:
- Spot rate
- Initial variance
- Model parameters (κ, θ, λ, ρ)
- Interest rates (domestic & foreign)

---

## How to Run

### Requirements

Install dependencies:

pip install numpy scipy matplotlib pandas

---

### Run the Script

python MATH398UpdatedV3.py

---

### Output

For each FX pair, the script generates:

- 3D volatility surface plots (PNG)
- CSV files with surface values

Saved to:

~/Desktop/

---

## Example Workflow

For each FX pair:

1. Build surface:

K, T, vol = build_surface(fx)

2. Plot surfaces:

plot_surface(K, T, vol, name, S0)
plot_surface_absolute(K, T, vol, name)

3. Save results:

save_surface_data(K, T, vol, name)
save_absolute_surface_data(K, T, vol, name)

---

## Notes

- Numerical derivatives use finite differences
- Implied volatility may return `NaN` if inversion fails
- Missing values are replaced with mean volatility during plotting
- Surfaces are smoothed via log-moneyness transformation

---

## Future Improvements

- Calibration to real market option data  
- Support for additional FX pairs  
- Faster numerical methods (vectorization)  
- Interactive plotting (Plotly)  
- Parallel computation for surface generation  
