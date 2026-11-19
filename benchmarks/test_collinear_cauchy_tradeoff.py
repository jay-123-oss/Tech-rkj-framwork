"""
Deep Empirical Analysis: Cauchy-Schwarz Bounding vs Feature Correlation Structure
in Ill-Conditioned and Strongly Collinear Settings (r = 0.90 to 0.9999).
"""

import numpy as np
from scipy import linalg
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tech_rkj import SubspaceRegressor


def run_collinear_benchmark():
    correlation_levels = [0.90, 0.99, 0.999, 0.9999]
    n_samples = 1000
    n_features = 12

    print(f"{'Correlation':<14} | {'Condition':<12} | {'Scipy OLS R2':<14} | {'Ridge R2':<12} | {'Subspace R2':<14} | {'Sub/Ridge Corr':<16} | {'Min Eig(P)':<12}")
    print("-" * 105)

    for r in correlation_levels:
        np.random.seed(42)
        # Latent factor decomposition to induce precise collinearity
        latent = np.random.randn(n_samples, 2)
        mixing = np.random.randn(2, n_features)

        # Scale noise to attain target pairwise correlation
        noise_std = np.sqrt(1.0 - r**2)
        X = latent @ mixing + noise_std * np.random.randn(n_samples, n_features)
        X = X.astype(np.float32)

        true_weights = np.random.uniform(-3.0, 3.0, size=n_features)
        y = (X @ true_weights + 0.1 * np.random.randn(n_samples)).astype(np.float64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Estimate empirical condition number of X_train^T X_train
        gram = X_train.T @ X_train
        eigvals_gram = np.linalg.eigvalsh(gram)
        cond_gram = np.max(eigvals_gram) / max(np.min(eigvals_gram), 1e-15)

        # 1. Scipy lstsq (unregularized OLS)
        ols_w, _, _, _ = linalg.lstsq(X_train, y_train)
        ols_preds = X_test @ ols_w
        ols_r2 = 1.0 - np.sum((y_test - ols_preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

        # 2. Sklearn Ridge (alpha = 1.0)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(X_train, y_train)
        ridge_preds = ridge.predict(X_test)
        ridge_r2 = ridge.score(X_test, y_test)

        # 3. SubspaceRegressor (alpha = 1.0)
        reg = SubspaceRegressor(
            num_heads=4,
            head_circuits=128,
            alpha=1.0,
            frontend_mode="identity",
        )
        reg.fit(X_train, y_train)
        sub_preds = reg.predict(X_test)
        sub_r2 = reg.score(X_test, y_test)

        # Pearson correlation between Ridge predictions and Subspace predictions
        pred_corr = np.corrcoef(ridge_preds, sub_preds)[0, 1]

        # Inspect minimum eigenvalue of P_inv across all heads
        min_eigs = []
        for h in range(reg.num_heads):
            p_mat = np.array(reg.p_inv_[h]).reshape(128, 128)
            eigs = np.linalg.eigvalsh(p_mat)
            min_eigs.append(np.min(eigs))
        overall_min_eig = min(min_eigs)

        print(
            f"r = {r:<10.4f} | {cond_gram:<12.2e} | {ols_r2:<14.6f} | {ridge_r2:<12.6f} | {sub_r2:<14.6f} | {pred_corr:<16.6f} | {overall_min_eig:<12.4e}"
        )

    print("\n" + "=" * 105)
    print("STREAMING CONTINUAL LEARNING STABILITY OVER 1,000 COLLINEAR SAMPLES (r = 0.999)")
    print("=" * 105)

    # Streaming test
    np.random.seed(1337)
    latent_stream = np.random.randn(1000, 2)
    mixing_stream = np.random.randn(2, n_features)
    X_stream = (latent_stream @ mixing_stream + 0.03 * np.random.randn(1000, n_features)).astype(np.float32)
    y_stream = (X_stream @ true_weights + 0.1 * np.random.randn(1000)).astype(np.float64)

    reg_stream = SubspaceRegressor(num_heads=4, head_circuits=128, alpha=1.0, frontend_mode="identity")
    reg_stream.fit(X_stream[:100], y_stream[:100])

    print(f"{'Stream Step':<15} | {'Test R2':<15} | {'Min Eigenvalue(P)':<20} | {'Max Eigenvalue(P)':<20} | {'Condition(P)':<15}")
    print("-" * 95)

    for step in range(100, 1000):
        reg_stream.update_online(X_stream[step], y_stream[step])
        if (step + 1) % 200 == 0 or step == 999:
            r2 = reg_stream.score(X_test, y_test)
            p_mat = np.array(reg_stream.p_inv_[0]).reshape(128, 128)
            eigs = np.linalg.eigvalsh(p_mat)
            min_e = np.min(eigs)
            max_e = np.max(eigs)
            cond_p = max_e / max(min_e, 1e-15)
            print(f"Step {step+1:<10} | {r2:<15.6f} | {min_e:<20.6e} | {max_e:<20.6e} | {cond_p:<15.2e}")


if __name__ == "__main__":
    run_collinear_benchmark()
