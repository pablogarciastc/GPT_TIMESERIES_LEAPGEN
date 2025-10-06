import numpy as np
import joblib
from scipy import stats
from scipy.signal import find_peaks
import json


class CompactPromptyDescriber:
    def __init__(self):
        pass

    def extract_compact_features(self, channel_data, channel_idx):
        """Extrae todas las características en UN SOLO string por variable en formato prompty"""

        # Calcular todas las características
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        median = np.median(channel_data)

        # Energía
        energy = np.sum(channel_data ** 2) / len(channel_data)
        rms = np.sqrt(energy)

        # Rango
        signal_range = np.ptp(channel_data)

        # Coeficiente de variación
        cv = std / (abs(mean) + 1e-9)

        # Tendencia
        slope, _, _, _, _ = stats.linregress(np.arange(len(channel_data)), channel_data)
        if abs(slope) < 0.01:
            trend = "no clear trend"
        elif slope > 0.05:
            trend = "strong upward trend"
        elif slope > 0:
            trend = "upward trend"
        elif slope < -0.05:
            trend = "strong downward trend"
        else:
            trend = "downward trend"

        # Volatilidad
        if cv < 0.05:
            volatility = "very low volatility"
        elif cv < 0.15:
            volatility = "low volatility"
        elif cv < 0.3:
            volatility = "moderate volatility"
        elif cv < 0.6:
            volatility = "high volatility"
        else:
            volatility = "very high volatility"

        # Autocorrelación
        if len(channel_data) > 1:
            autocorr1 = np.corrcoef(channel_data[:-1], channel_data[1:])[0, 1]
            autocorr1 = autocorr1 if not np.isnan(autocorr1) else 0.0
        else:
            autocorr1 = 0.0

        # Periodicidad
        dominant_period, period_strength = self._find_dominant_period(channel_data)
        if dominant_period > 0 and period_strength > 0.7:
            seasonality = f"strong seasonality with dominant period {dominant_period}"
        elif dominant_period > 0 and period_strength > 0.4:
            seasonality = f"moderate seasonality with dominant period {dominant_period}"
        elif dominant_period > 0 and period_strength > 0.2:
            seasonality = f"weak seasonality with dominant period {dominant_period}"
        else:
            seasonality = "no seasonality detected"

        # Outliers
        q1, q3 = np.percentile(channel_data, [25, 75])
        iqr = q3 - q1
        outliers = ((channel_data < q1 - 1.5 * iqr) | (channel_data > q3 + 1.5 * iqr)).sum()
        if outliers > 0:
            outlier_text = f"{outliers} outliers detected"
        else:
            outlier_text = "no outliers"

        # Skewness y Kurtosis
        skewness = stats.skew(channel_data)
        kurtosis = stats.kurtosis(channel_data)

        # Picos
        if std > 1e-6:
            normalized = (channel_data - mean) / std
            peaks, _ = find_peaks(normalized, prominence=0.5, distance=3)
            peak_freq = len(peaks) / len(channel_data)
        else:
            peak_freq = 0.0
            peaks = []

        # Cambio medio
        if len(channel_data) > 1:
            mean_abs_change = np.mean(np.abs(np.diff(channel_data)))
        else:
            mean_abs_change = 0.0

        # Construir el string prompty
        prompty_str = (
            f"variable {channel_idx}: "
            f"mean value around {mean:.3f}, "
            f"standard deviation of {std:.3f}, "
            f"median {median:.3f}, "
            f"energy {energy:.3f}, "
            f"RMS {rms:.3f}, "
            f"range {signal_range:.3f}, "
            f"{trend}, "
            f"{volatility}, "
            f"autocorrelation at lag 1 {autocorr1:.3f}, "
            f"{seasonality}, "
            f"{outlier_text}, "
            f"skewness {skewness:.3f}, "
            f"kurtosis {kurtosis:.3f}, "
            f"{len(peaks)} peaks detected, "
            f"mean absolute change {mean_abs_change:.3f}, "
            f"IQR {iqr:.3f}"
        )

        return prompty_str

    def _find_dominant_period(self, ts):
        """Encuentra el periodo dominante usando autocorrelación"""
        max_lag = min(len(ts) // 3, 50)
        if max_lag < 2:
            return 0, 0.0

        autocorrs = []
        for lag in range(2, max_lag):
            if lag >= len(ts):
                break
            corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
            if not np.isnan(corr) and not np.isinf(corr):
                autocorrs.append((lag, abs(corr)))

        if autocorrs:
            autocorrs.sort(key=lambda x: x[1], reverse=True)
            return autocorrs[0][0], autocorrs[0][1]
        return 0, 0.0

    def describe_class(self, X_class):
        """Genera descripción compacta de todos los canales para una clase"""
        n_samples, n_timesteps, n_channels = X_class.shape

        # Promedio entre muestras para cada canal
        avg_per_channel = np.mean(X_class, axis=0)  # (timesteps, channels)

        # Extraer características de cada canal en UN string
        descriptions = []
        for ch_idx in range(n_channels):
            desc = self.extract_compact_features(avg_per_channel[:, ch_idx], ch_idx)
            descriptions.append(desc)

        return descriptions

    def describe_dataset(self, X, y):
        """Genera descripciones para todas las clases"""
        classes = np.unique(y)
        dataset_descriptions = {}

        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(classes)}")
        print(f"Generating compact descriptions...\n")

        for cls in classes:
            X_class = X[y == cls]
            print(f"Processing class {cls} ({len(X_class)} samples, {X_class.shape[2]} channels)...")

            descriptions = self.describe_class(X_class)
            dataset_descriptions[str(cls)] = descriptions

        return dataset_descriptions


if __name__ == "__main__":
    dataset_name = "DailySports"

    # Cargar datos
    X_train = joblib.load("data/" + dataset_name + "/x_train.pkl")
    y_train = joblib.load("data/" + dataset_name + "/state_train.pkl")

    print(f"Loaded data with shape: {X_train.shape}\n")

    # Crear descripciones compactas
    describer = CompactPromptyDescriber()
    descriptions = describer.describe_dataset(X_train, y_train)

    # Nombres de actividades
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

    # Renombrar con nombres de actividades
    descriptions_renamed = {}
    for old_key, desc_list in descriptions.items():
        new_key = activity_names.get(old_key, f"class_{old_key}")
        descriptions_renamed[new_key] = desc_list

    # Guardar
    output_path = "per_channel_" + dataset_name + ".json"
    with open(output_path, "w") as f:
        json.dump(descriptions_renamed, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Compact prompty descriptions saved to {output_path}")
    print(f"{'=' * 70}")

    # Mostrar ejemplo
    print("\nExample - First activity, first 3 variables:\n")
    first_activity = list(descriptions_renamed.keys())[0]
    print(f"{first_activity}:")
    for i in range(min(3, len(descriptions_renamed[first_activity]))):
        print(f"  {descriptions_renamed[first_activity][i]}")

    print(f"\n  ... (total {len(descriptions_renamed[first_activity])} variables)")

    # Estadística del archivo
    import os

    file_size = os.path.getsize(output_path)
    print(f"\nFile size: {file_size / 1024:.2f} KB")