# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(10)
X1 = np.random.normal([-1, -1], 1, (1000, 2))
X2 = np.random.normal([1, 1], 1, (1000, 2))

X = np.concatenate([X1, X2])
y = np.concatenate([np.zeros(1000), np.ones(1000)])

plt.scatter(X[:, 0], X[:, 1], c=y)

X_scaled = X.copy()
X_scaled[:, 0] *= 10

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)

rfc1 = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)
rfc1.fit(X, y)
yhat1 = rfc1.predict(X)

rfc2 = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)
rfc2.fit(X_scaled, y)
yhat2 = rfc2.predict(X_scaled)

np.all(yhat1 == yhat2)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=yhat2)
