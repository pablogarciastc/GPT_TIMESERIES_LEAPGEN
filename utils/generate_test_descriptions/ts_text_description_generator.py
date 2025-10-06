import numpy as np
import joblib
from scipy import stats
import json

class LEAPGENStyleDescriber:
    def __init__(self):
        pass

    def extract_features(self, ts: np.ndarray):
        feats = []
        mean = np.mean(ts)
        std = np.std(ts)
        feats.append(f"mean value around {mean:.3f}")
        feats.append(f"standard deviation of {std:.3f}")

        slope, _, _, _, _ = stats.linregress(np.arange(len(ts)), ts)
        if abs(slope) < 0.01:
            feats.append("no clear trend")
        elif slope > 0:
            feats.append("upward trend")
        else:
            feats.append("downward trend")

        cv = std / (abs(mean) + 1e-9)
        if cv < 0.05:
            feats.append("very low volatility")
        elif cv < 0.1:
            feats.append("low volatility")
        elif cv < 0.2:
            feats.append("moderate volatility")
        else:
            feats.append("high volatility")

        max_lag = min(len(ts) // 2, 50)
        autocorrs = []
        for lag in range(1, max_lag):
            corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append((lag, abs(corr)))
        if autocorrs:
            autocorrs.sort(key=lambda x: x[1], reverse=True)
            dominant_lag, max_corr = autocorrs[0]
            if max_corr > 0.7:
                feats.append(f"strong seasonality with dominant period {dominant_lag}")
            elif max_corr > 0.3:
                feats.append(f"moderate seasonality with dominant period {dominant_lag}")
            else:
                feats.append("weak seasonality")
        else:
            feats.append("no seasonality detected")

        q1, q3 = np.percentile(ts, [25, 75])
        iqr = q3 - q1
        outliers = ((ts < q1 - 1.5 * iqr) | (ts > q3 + 1.5 * iqr)).sum()
        feats.append(f"{outliers} outliers detected" if outliers > 0 else "no outliers")

        feats.append(f"skewness {stats.skew(ts):.3f}")
        feats.append(f"kurtosis {stats.kurtosis(ts):.3f}")

        if len(ts) > 1:
            autocorr1 = np.corrcoef(ts[:-1], ts[1:])[0, 1]
            feats.append(f"autocorrelation at lag 1 {autocorr1:.3f}")
        return feats

    def describe_dataset(self, X, y):
        classes = np.unique(y)
        class_descriptions = {}

        for cls in classes:
            X_class = X[y == cls]
            combined_ts = np.mean(X_class, axis=0)  # promedio entre muestras

            if combined_ts.ndim > 1:
                combined_ts = np.mean(combined_ts, axis=1)  # promedio entre canales

            features = self.extract_features(combined_ts)
            class_descriptions[str(cls)] = features

        return class_descriptions


if __name__ == "__main__":

    dataset_name = "DailySports"

    X_train = joblib.load("data/" + dataset_name +  "/x_train.pkl")
    y_train = joblib.load("data/" + dataset_name + "/state_train.pkl")

    describer = LEAPGENStyleDescriber()
    descriptions = describer.describe_dataset(X_train, y_train)

    activity_names = {
        '0.0': 'Sitting',
        '1.0': 'Standing',
        '2.0': 'Lying on back',
        '3.0': 'Lying on right side',
        '4.0': 'Ascending stairs',
        '5.0': 'Descending stairs',
        '6.0': 'Standing in elevator still',
        '7.0': 'Moving around in elevator',
        '8.0': 'Walking in parking lot',
        '9.0': 'Walking on treadmill flat',
        '10.0': 'Walking on treadmill inclined',
        '11.0': 'Running on treadmill',
        '12.0': 'Exercising on stepper',
        '13.0': 'Exercising on cross trainer',
        '14.0': 'Cycling horizontal',
        '15.0': 'Cycling vertical',
        '16.0': 'Rowing',
        '17.0': 'Jumping'
    }

    # Crear nuevo diccionario con nombres de actividades como claves
    descriptions_renamed = {}

    for old_key, description_list in descriptions.items():
        new_key = activity_names.get(old_key, f"Unknown_activity_{old_key}")
        descriptions_renamed[new_key] = description_list

    with open("../../descriptors/descriptors_" + dataset_name +  ".json", "w") as f:
        json.dump(descriptions_renamed, f, indent=4)

    print("JSON descriptions saved to dataset_descriptions.json")
