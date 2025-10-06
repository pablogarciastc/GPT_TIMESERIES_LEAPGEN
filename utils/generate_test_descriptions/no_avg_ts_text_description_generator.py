import numpy as np
import joblib
from scipy import stats
from scipy.signal import find_peaks
import json


class AgnosticMultiChannelDescriber:
    def __init__(self):
        pass

    def extract_channel_features(self, channel_data):
        """Extrae características estadísticas de un canal"""
        features = {}

        # Estadísticas básicas
        features['mean'] = np.mean(channel_data)
        features['std'] = np.std(channel_data)
        features['median'] = np.median(channel_data)
        features['range'] = np.ptp(channel_data)
        features['iqr'] = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)

        # Coeficiente de variación
        features['cv'] = features['std'] / (abs(features['mean']) + 1e-9)

        # Energía de la señal
        features['energy'] = np.sum(channel_data ** 2) / len(channel_data)
        features['rms'] = np.sqrt(features['energy'])

        # Tendencia
        slope, _, _, _, _ = stats.linregress(np.arange(len(channel_data)), channel_data)
        features['slope'] = slope

        # Autocorrelación lag-1
        if len(channel_data) > 1:
            features['autocorr1'] = np.corrcoef(channel_data[:-1], channel_data[1:])[0, 1]
            if np.isnan(features['autocorr1']):
                features['autocorr1'] = 0
        else:
            features['autocorr1'] = 0

        # Detección de picos
        if features['std'] > 0:
            normalized = (channel_data - features['mean']) / features['std']
            peaks, _ = find_peaks(normalized, prominence=0.5, distance=3)
            features['peak_count'] = len(peaks)
            features['peak_frequency'] = len(peaks) / len(channel_data)
        else:
            features['peak_count'] = 0
            features['peak_frequency'] = 0

        # Periodicidad dominante
        features['dominant_period'] = self._find_dominant_period(channel_data)

        # Forma de distribución
        features['skewness'] = stats.skew(channel_data)
        features['kurtosis'] = stats.kurtosis(channel_data)

        # Outliers
        q1, q3 = np.percentile(channel_data, [25, 75])
        iqr = q3 - q1
        outliers = ((channel_data < q1 - 1.5 * iqr) | (channel_data > q3 + 1.5 * iqr)).sum()
        features['outlier_ratio'] = outliers / len(channel_data)

        return features

    def _find_dominant_period(self, ts):
        """Encuentra el periodo dominante usando autocorrelación"""
        max_lag = min(len(ts) // 3, 40)
        if max_lag < 2:
            return 0

        autocorrs = []
        for lag in range(2, max_lag):
            if lag >= len(ts):
                break
            corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
            if not np.isnan(corr) and not np.isinf(corr):
                autocorrs.append((lag, abs(corr)))

        if autocorrs:
            autocorrs.sort(key=lambda x: x[1], reverse=True)
            if autocorrs[0][1] > 0.3:
                return autocorrs[0][0]
        return 0

    def analyze_channels(self, X_class):
        """Analiza todos los canales y extrae estadísticas agregadas"""
        n_samples, n_timesteps, n_channels = X_class.shape

        # Promedio entre muestras para cada canal
        avg_per_channel = np.mean(X_class, axis=0)  # (timesteps, channels)

        # Extraer características de cada canal
        channel_features = []
        for ch_idx in range(n_channels):
            feats = self.extract_channel_features(avg_per_channel[:, ch_idx])
            channel_features.append(feats)

        # Agregar características cross-channel
        cross_features = self._compute_cross_channel_features(channel_features, avg_per_channel)

        return channel_features, cross_features

    def _compute_cross_channel_features(self, channel_features, avg_per_channel):
        """Calcula características entre canales"""
        n_channels = len(channel_features)

        # Distribución de energía entre canales
        energies = np.array([cf['energy'] for cf in channel_features])
        energy_percentiles = np.percentile(energies, [25, 50, 75, 90])

        # Canales más activos
        top_channels_idx = np.argsort(energies)[-min(5, n_channels):][::-1]

        # Correlación entre canales
        if avg_per_channel.shape[1] > 1:
            corr_matrix = np.corrcoef(avg_per_channel.T)
            np.fill_diagonal(corr_matrix, np.nan)
            avg_corr = np.nanmean(np.abs(corr_matrix))
            max_corr = np.nanmax(np.abs(corr_matrix))
        else:
            avg_corr = 0
            max_corr = 0

        # Variabilidad entre canales
        stds = np.array([cf['std'] for cf in channel_features])
        cv_across_channels = np.std(stds) / (np.mean(stds) + 1e-9)

        # Periodicidad común
        periods = [cf['dominant_period'] for cf in channel_features if cf['dominant_period'] > 0]
        common_periodicity = len(periods) / n_channels if n_channels > 0 else 0

        return {
            'energy_p25': energy_percentiles[0],
            'energy_median': energy_percentiles[1],
            'energy_p75': energy_percentiles[2],
            'energy_p90': energy_percentiles[3],
            'top_channels': top_channels_idx.tolist(),
            'avg_channel_correlation': avg_corr,
            'max_channel_correlation': max_corr,
            'cv_across_channels': cv_across_channels,
            'common_periodicity_ratio': common_periodicity,
            'avg_period': np.mean(periods) if periods else 0
        }

    def describe_activity(self, X_class):
        """Genera descripción textual de una clase"""
        channel_features, cross_features = self.analyze_channels(X_class)
        descriptions = []

        # 1. Nivel de actividad general
        energy_median = cross_features['energy_median']
        energy_p90 = cross_features['energy_p90']

        if energy_p90 > 100:
            descriptions.append("high overall activity level")
        elif energy_p90 > 50:
            descriptions.append("moderate overall activity level")
        elif energy_p90 > 10:
            descriptions.append("low activity level")
        else:
            descriptions.append("minimal/static activity")

        # 2. Distribución de actividad entre canales
        n_channels = len(channel_features)
        active_channels = sum(1 for cf in channel_features if cf['energy'] > energy_median)
        active_ratio = active_channels / n_channels

        if active_ratio > 0.7:
            descriptions.append(f"widespread activation across channels ({active_channels}/{n_channels})")
        elif active_ratio > 0.4:
            descriptions.append(f"selective channel activation ({active_channels}/{n_channels})")
        else:
            descriptions.append(f"localized activation in few channels ({active_channels}/{n_channels})")

        # 3. Canales dominantes
        top_ch = cross_features['top_channels'][:3]
        descriptions.append(f"primary channels: {', '.join([f'ch{i}' for i in top_ch])}")

        # 4. Coordinación entre canales
        if cross_features['avg_channel_correlation'] > 0.7:
            descriptions.append("highly coordinated multi-channel pattern")
        elif cross_features['avg_channel_correlation'] > 0.4:
            descriptions.append("moderate inter-channel coordination")
        else:
            descriptions.append("independent channel dynamics")

        # 5. Variabilidad temporal
        avg_std = np.mean([cf['std'] for cf in channel_features])
        if avg_std > 5:
            descriptions.append("high temporal variability")
        elif avg_std > 2:
            descriptions.append("moderate temporal variability")
        else:
            descriptions.append("stable/consistent temporal pattern")

        # 6. Patrones rítmicos
        if cross_features['common_periodicity_ratio'] > 0.5:
            avg_period = cross_features['avg_period']
            descriptions.append(
                f"rhythmic pattern (period ~{avg_period:.1f} timesteps) in {cross_features['common_periodicity_ratio'] * 100:.0f}% of channels")
        elif cross_features['common_periodicity_ratio'] > 0.2:
            descriptions.append("partial rhythmic behavior in some channels")
        else:
            descriptions.append("non-periodic/irregular pattern")

        # 7. Heterogeneidad entre canales
        if cross_features['cv_across_channels'] > 1.0:
            descriptions.append("highly heterogeneous channel behavior")
        elif cross_features['cv_across_channels'] > 0.5:
            descriptions.append("moderate heterogeneity across channels")
        else:
            descriptions.append("homogeneous channel behavior")

        # 8. Características de picos
        avg_peak_freq = np.mean([cf['peak_frequency'] for cf in channel_features])
        if avg_peak_freq > 0.15:
            descriptions.append("frequent peaks/rapid changes")
        elif avg_peak_freq > 0.08:
            descriptions.append("moderate peak frequency")
        else:
            descriptions.append("smooth/gradual changes")

        # 9. Tendencias
        slopes = [cf['slope'] for cf in channel_features]
        avg_slope = np.mean(slopes)
        if abs(avg_slope) > 0.05:
            trend = "upward" if avg_slope > 0 else "downward"
            descriptions.append(f"{trend} trend across channels")
        else:
            descriptions.append("no clear directional trend")

        # 10. Distribución de valores
        avg_skew = np.mean([cf['skewness'] for cf in channel_features])
        if abs(avg_skew) > 1.0:
            skew_dir = "right" if avg_skew > 0 else "left"
            descriptions.append(f"{skew_dir}-skewed distribution")

        return descriptions

    def describe_dataset(self, X, y):
        """Genera descripciones para todas las clases"""
        classes = np.unique(y)
        class_descriptions = {}

        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(classes)}")

        for cls in classes:
            X_class = X[y == cls]
            print(f"\nProcessing class {cls} ({len(X_class)} samples)...")
            descriptions = self.describe_activity(X_class)
            class_descriptions[str(cls)] = descriptions

        return class_descriptions


if __name__ == "__main__":
    dataset_name = "DailySports"

    # Cargar datos
    X_train = joblib.load("data/" + dataset_name + "/x_train.pkl")
    y_train = joblib.load("data/" + dataset_name + "/state_train.pkl")

    print(f"Loaded data with shape: {X_train.shape}")

    # Crear descripciones
    describer = AgnosticMultiChannelDescriber()
    descriptions = describer.describe_dataset(X_train, y_train)

    # Nombres de actividades (específico del dataset, pero opcional)
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

    # Renombrar con nombres de actividades si existen
    descriptions_renamed = {}
    for old_key, description_list in descriptions.items():
        new_key = activity_names.get(old_key, f"class_{old_key}")
        descriptions_renamed[new_key] = description_list

    # Guardar
    output_path = "no_avg_descriptors_" + dataset_name + ".json"
    with open(output_path, "w") as f:
        json.dump(descriptions_renamed, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"Descriptions saved to {output_path}")
    print(f"{'=' * 60}")

    # Mostrar ejemplos
    print("\nExample descriptions:\n")
    for activity, desc in list(descriptions_renamed.items())[:3]:
        print(f"{activity}:")
        for d in desc:
            print(f"  • {d}")
        print()