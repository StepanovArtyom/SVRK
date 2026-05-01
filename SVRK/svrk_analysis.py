"""
Исследование возможности прогнозирования состояния реакторной установки ВВЭР-1000 
по показаниям СВРК с применением методов машинного обучения

Данный модуль реализует полный цикл обработки данных:
1. Предобработка данных (удаление константных признаков, интерполяция, сглаживание, нормализация)
2. Кластеризация временных точек методом k-средних для выделения устойчивых режимов
3. PCA для снижения размерности внутри каждого кластера
4. Прогнозирование тепловой мощности реактора с помощью LSTM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data(data_path, params_path=None):
    """
    Загрузка данных из Excel файлов
    
    Parameters:
    -----------
    data_path : str - путь к файлу с данными измерений
    params_path : str - путь к файлу с наименованиями параметров (опционально)
    
    Returns:
    --------
    df : DataFrame - загруженные данные
    param_names : dict - словарь с наименованиями параметров (если предоставлен)
    """
    print(f"Загрузка данных из {data_path}...")
    df = pd.read_excel(data_path)
    
    # Преобразование первого столбца в datetime и установка в качестве индекса
    first_col = df.columns[0]
    df[first_col] = pd.to_datetime(df[first_col])
    df.set_index(first_col, inplace=True)
    
    param_names = None
    if params_path:
        print(f"Загрузка наименований параметров из {params_path}...")
        param_df = pd.read_excel(params_path)
        param_names = dict(zip(param_df.iloc[:, 0], param_df.iloc[:, 1]))
    
    print(f"Загружено {len(df)} записей с {len(df.columns)} параметрами")
    print(f"Период данных: {df.index.min()} - {df.index.max()}")
    
    return df, param_names


def remove_constant_features(df, threshold=0.99):
    """
    Удаление константных и почти константных признаков
    
    Parameters:
    -----------
    df : DataFrame - исходные данные
    threshold : float - порог для определения константности (доля одинаковых значений)
    
    Returns:
    --------
    df_clean : DataFrame - данные без константных признаков
    removed_features : list - список удаленных признаков
    """
    print("\n=== Этап 1.1: Удаление константных признаков ===")
    
    removed_features = []
    columns_to_keep = []
    
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > (1 - threshold):
            removed_features.append(col)
        else:
            columns_to_keep.append(col)
    
    df_clean = df[columns_to_keep].copy()
    
    print(f"Удалено {len(removed_features)} константных признаков:")
    if len(removed_features) <= 10:
        print(f"  {removed_features}")
    else:
        print(f"  {removed_features[:10]} ... и еще {len(removed_features)-10}")
    print(f"Осталось признаков: {len(columns_to_keep)}")
    
    return df_clean, removed_features


def interpolate_missing_values(df, method='linear'):
    """
    Интерполяция пропущенных значений
    
    Parameters:
    -----------
    df : DataFrame - данные с пропусками
    method : str - метод интерполяции ('linear', 'polynomial', 'spline')
    
    Returns:
    --------
    df_interp : DataFrame - данные с интерполированными пропусками
    """
    print("\n=== Этап 1.2: Интерполяция пропусков ===")
    
    missing_before = df.isnull().sum().sum()
    
    df_interp = df.interpolate(method=method, limit_direction='both')
    
    # Заполнение оставшихся NaN на краях
    df_interp = df_interp.fillna(method='bfill').fillna(method='ffill')
    
    missing_after = df_interp.isnull().sum().sum()
    
    print(f"Пропусков до интерполяции: {missing_before}")
    print(f"Пропусков после интерполяции: {missing_after}")
    
    return df_interp


def smooth_data(df, window_size=5):
    """
    Сглаживание данных скользящим средним
    
    Parameters:
    -----------
    df : DataFrame - исходные данные
    window_size : int - размер окна скользящего среднего
    
    Returns:
    --------
    df_smooth : DataFrame - сглаженные данные
    """
    print("\n=== Этап 1.3: Сглаживание скользящим средним ===")
    print(f"Размер окна сглаживания: {window_size}")
    
    df_smooth = df.rolling(window=window_size, center=True).mean()
    
    # Заполнение краев
    df_smooth = df_smooth.fillna(method='bfill').fillna(method='ffill')
    
    print(f"Данные сглажены с окном {window_size} точек")
    
    return df_smooth


def normalize_data(df):
    """
    Нормализация данных (StandardScaler)
    
    Parameters:
    -----------
    df : DataFrame - исходные данные
    
    Returns:
    --------
    df_norm : DataFrame - нормализованные данные
    scaler : StandardScaler - обученный скалер для обратного преобразования
    """
    print("\n=== Этап 1.4: Нормализация данных ===")
    
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)
    df_norm = pd.DataFrame(df_normalized, index=df.index, columns=df.columns)
    
    print(f"Данные нормализованы (среднее ~0, std ~1)")
    print(f"Среднее значение после нормализации: {df_norm.mean().mean():.6f}")
    print(f"Стандартное отклонение после нормализации: {df_norm.std().mean():.6f}")
    
    return df_norm, scaler


def preprocess_data(df, window_size=5):
    """
    Полный цикл предобработки данных
    
    Parameters:
    -----------
    df : DataFrame - исходные данные
    window_size : int - размер окна для сглаживания
    
    Returns:
    --------
    df_processed : DataFrame - полностью обработанные данные
    preprocessing_info : dict - информация о предобработке
    """
    print("\n" + "="*60)
    print("ЭТАП 1: ПРЕДОБРАБОТКА ДАННЫХ")
    print("="*60)
    
    # Удаление константных признаков
    df_clean, removed_features = remove_constant_features(df)
    
    # Интерполяция пропусков
    df_interp = interpolate_missing_values(df_clean)
    
    # Сглаживание
    df_smooth = smooth_data(df_interp, window_size=window_size)
    
    # Нормализация
    df_norm, scaler = normalize_data(df_smooth)
    
    preprocessing_info = {
        'removed_features': removed_features,
        'scaler': scaler,
        'window_size': window_size
    }
    
    print("\n" + "="*60)
    print("ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
    print("="*60)
    
    return df_norm, preprocessing_info


def find_optimal_clusters(X, max_k=10, random_state=42):
    """
    Поиск оптимального числа кластеров по силуэтному коэффициенту
    
    Parameters:
    -----------
    X : array-like - данные для кластеризации
    max_k : int - максимальное число кластеров для проверки
    random_state : int - seed для воспроизводимости
    
    Returns:
    --------
    optimal_k : int - оптимальное число кластеров
    silhouette_scores : dict - силуэтные коэффициенты для разных k
    """
    print("\n=== Поиск оптимального числа кластеров ===")
    
    silhouette_scores = {}
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores[k] = score
        print(f"k={k}: силуэтный коэффициент = {score:.4f}")
    
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\nОптимальное число кластеров: k={optimal_k} (silhouette={silhouette_scores[optimal_k]:.4f})")
    
    return optimal_k, silhouette_scores


def cluster_analysis(df_processed, optimal_k=None, random_state=42):
    """
    Кластеризация временных точек методом k-средних
    
    Parameters:
    -----------
    df_processed : DataFrame - предобработанные данные
    optimal_k : int - число кластеров (если None, подбирается автоматически)
    random_state : int - seed для воспроизводимости
    
    Returns:
    --------
    labels : array - метки кластеров для каждой точки
    kmeans : KMeans - обученная модель кластеризации
    cluster_info : dict - информация о кластерах
    """
    print("\n" + "="*60)
    print("ЭТАП 2: КЛАСТЕРИЗАЦИЯ ВРЕМЕННЫХ ТОЧЕК")
    print("="*60)
    
    X = df_processed.values
    
    # Поиск оптимального числа кластеров если не задано
    if optimal_k is None:
        optimal_k, silhouette_scores = find_optimal_clusters(X, max_k=10, random_state=random_state)
    else:
        silhouette_scores = {}
        for k in range(2, 11):
            kmeans_temp = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X)
            silhouette_scores[k] = silhouette_score(X, labels_temp)
    
    # Финальная кластеризация
    print(f"\nКластеризация с k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Анализ распределения по кластерам
    unique, counts = np.unique(labels, return_counts=True)
    print("\nРаспределение точек по кластерам:")
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(labels) * 100
        print(f"  Кластер {cluster_id}: {count} точек ({percentage:.1f}%)")
    
    # Вычисление силуэтного коэффициента
    final_silhouette = silhouette_score(X, labels)
    print(f"\nИтоговый силуэтный коэффициент: {final_silhouette:.4f}")
    
    cluster_info = {
        'n_clusters': optimal_k,
        'silhouette_score': final_silhouette,
        'cluster_sizes': dict(zip(unique, counts)),
        'silhouette_scores_all': silhouette_scores
    }
    
    print("\n" + "="*60)
    print("КЛАСТЕРИЗАЦИЯ ЗАВЕРШЕНА")
    print("="*60)
    
    return labels, kmeans, cluster_info


def analyze_cluster_profiles(df_original, labels, param_names=None):
    """
    Анализ средних профилей параметров по кластерам
    
    Parameters:
    -----------
    df_original : DataFrame - исходные (не нормализованные) данные
    labels : array - метки кластеров
    param_names : dict - словарь с наименованиями параметров
    
    Returns:
    --------
    cluster_profiles : DataFrame - средние значения параметров по кластерам
    """
    print("\n=== Анализ средних профилей кластеров ===")
    
    df_with_labels = df_original.copy()
    df_with_labels['cluster'] = labels
    
    cluster_profiles = df_with_labels.groupby('cluster').mean()
    
    print("\nСредние профили кластеров (первые 10 параметров):")
    print(cluster_profiles.iloc[:, :10])
    
    return cluster_profiles


def apply_pca_per_cluster(df_processed, labels, variance_threshold=0.95, random_state=42):
    """
    Применение PCA для каждого кластера отдельно
    
    Parameters:
    -----------
    df_processed : DataFrame - предобработанные данные
    labels : array - метки кластеров
    variance_threshold : float - порог сохраняемой дисперсии
    random_state : int - seed для воспроизводимости
    
    Returns:
    --------
    pca_results : dict - результаты PCA для каждого кластера
    """
    print("\n" + "="*60)
    print("ЭТАП 3: PCA ДЛЯ КАЖДОГО КЛАСТЕРА")
    print("="*60)
    
    X = df_processed.values
    unique_clusters = np.unique(labels)
    
    pca_results = {}
    
    for cluster_id in unique_clusters:
        print(f"\n--- Кластер {cluster_id} ---")
        
        # Выбор точек belonging к кластеру
        cluster_mask = labels == cluster_id
        X_cluster = X[cluster_mask]
        
        print(f"Количество точек в кластере: {X_cluster.shape[0]}")
        
        # Определение числа компонент для сохранения заданной дисперсии
        n_components_max = min(X_cluster.shape[0] - 1, X_cluster.shape[1])
        
        if n_components_max < 2:
            print(f"Предупреждение: слишком мало точек в кластере {cluster_id}")
            continue
        
        pca_full = PCA(random_state=random_state)
        pca_full.fit(X_cluster)
        
        # Поиск числа компонент для заданного порога дисперсии
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        n_components = min(n_components, n_components_max)
        
        # Финальный PCA с выбранным числом компонент
        pca = PCA(n_components=n_components, random_state=random_state)
        X_pca = pca.fit_transform(X_cluster)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        
        print(f"Число компонент PCA: {n_components}")
        print(f"Сохранённая дисперсия: {explained_variance:.4f} ({explained_variance*100:.1f}%)")
        print(f"Исходная размерность: {X_cluster.shape[1]} -> Новая размерность: {X_pca.shape[1]}")
        
        pca_results[cluster_id] = {
            'pca_model': pca,
            'X_pca': X_pca,
            'n_components': n_components,
            'explained_variance': explained_variance,
            'indices': np.where(cluster_mask)[0]
        }
    
    print("\n" + "="*60)
    print("PCA ЗАВЕРШЕН")
    print("="*60)
    
    return pca_results


def visualize_clusters_and_pca(df_processed, labels, pca_results, save_dir=None):
    """
    Визуализация кластеров и результатов PCA
    
    Parameters:
    -----------
    df_processed : DataFrame - предобработанные данные
    labels : array - метки кластеров
    pca_results : dict - результаты PCA
    save_dir : str - директория для сохранения графиков
    """
    print("\n=== Визуализация результатов ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Распределение кластеров во времени
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
    ax.scatter(range(len(labels)), [0]*len(labels), c=colors[labels], s=10, alpha=0.6)
    ax.set_xlabel('Временная точка')
    ax.set_ylabel('')
    ax.set_title('Распределение кластеров во времени')
    ax.set_yticks([])
    
    # 2. Силуэтный коэффициент для разных k
    ax = axes[0, 1]
    # Эта информация должна быть передана дополнительно
    
    # 3-4. PCA проекции для первых двух кластеров
    unique_clusters = sorted(pca_results.keys())
    
    for idx, cluster_id in enumerate(unique_clusters[:2]):
        ax = axes[1, idx] if len(unique_clusters) >= 2 else axes[1, 0]
        pca_data = pca_results[cluster_id]['X_pca']
        
        if pca_data.shape[1] >= 2:
            ax.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.5, s=20)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'Кластер {cluster_id}: PCA проекция\n(дисперсия: {pca_results[cluster_id]["explained_variance"]:.2%})')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/clusters_pca_visualization.png", dpi=150, bbox_inches='tight')
        print(f"График сохранён в {save_dir}/clusters_pca_visualization.png")
    
    plt.show()


def prepare_lstm_data(df_processed, labels, pca_results, target_col='NAKЗ', 
                      history_window=10, forecast_horizon=1, test_ratio=0.2):
    """
    Подготовка данных для LSTM модели
    
    Parameters:
    -----------
    df_processed : DataFrame - предобработанные данные
    labels : array - метки кластеров
    pca_results : dict - результаты PCA
    target_col : str - имя целевой переменной (тепловая мощность)
    history_window : int - длина окна истории (часов)
    forecast_horizon : int - горизонт прогноза (часов)
    test_ratio : float - доля тестовой выборки
    
    Returns:
    --------
    X_train, y_train, X_val, y_val : arrays - обучающая и валидационная выборки
    scalers : dict - скалеры для целевой переменной
    """
    print("\n" + "="*60)
    print("ЭТАП 4: ПОДГОТОВКА ДАННЫХ ДЛЯ LSTM")
    print("="*60)
    
    # Проверка наличия целевой переменной
    if target_col not in df_processed.columns:
        print(f"Предупреждение: колонка '{target_col}' не найдена.")
        print("Доступные колонки:", df_processed.columns[:10].tolist(), "...")
        # Используем первую числовую колонку как целевую
        target_col = df_processed.select_dtypes(include=[np.number]).columns[0]
        print(f"Используем '{target_col}' как целевую переменную")
    
    # Создание единого представления PCA компонент
    n_samples = len(df_processed)
    n_pca_features = sum([pca_results[c]['n_components'] for c in pca_results])
    
    X_pca_combined = np.zeros((n_samples, n_pca_features))
    
    # Заполнение матрицы PCA компонентами
    feature_idx = 0
    for cluster_id in sorted(pca_results.keys()):
        indices = pca_results[cluster_id]['indices']
        X_pca = pca_results[cluster_id]['X_pca']
        n_comp = X_pca.shape[1]
        X_pca_combined[indices, feature_idx:feature_idx+n_comp] = X_pca
        feature_idx += n_comp
    
    print(f"Общая размерность PCA признаков: {n_pca_features}")
    
    # Целевая переменная
    y_full = df_processed[target_col].values
    
    # Скалирование целевой переменной
    from sklearn.preprocessing import MinMaxScaler
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y_full.reshape(-1, 1)).flatten()
    
    # Создание последовательностей для LSTM
    def create_sequences(X, y, window, horizon):
        X_seq, y_seq = [], []
        for i in range(len(X) - window - horizon + 1):
            X_seq.append(X[i:i+window])
            y_seq.append(y[i+window+horizon-1])
        return np.array(X_seq), np.array(y_seq)
    
    X_seq, y_seq = create_sequences(X_pca_combined, y_scaled, history_window, forecast_horizon)
    
    print(f"Форма последовательностей: X={X_seq.shape}, y={y_seq.shape}")
    
    # Разделение на train/val (80/20)
    split_idx = int(len(X_seq) * 0.8)
    
    X_train = X_seq[:split_idx]
    y_train = y_seq[:split_idx]
    X_val = X_seq[split_idx:]
    y_val = y_seq[split_idx:]
    
    print(f"\nРазмер обучающей выборки: {X_train.shape[0]} примеров")
    print(f"Размер валидационной выборки: {X_val.shape[0]} примеров")
    
    scalers = {'y': y_scaler}
    
    print("\n" + "="*60)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ LSTM ЗАВЕРШЕНА")
    print("="*60)
    
    return X_train, y_train, X_val, y_val, scalers


def build_lstm_model(input_shape, output_shape=1, units=64, dropout=0.2):
    """
    Построение LSTM модели
    
    Parameters:
    -----------
    input_shape : tuple - форма входных данных (window, features)
    output_shape : int - размерность выхода
    units : int - число единиц в LSTM слое
    dropout : float - коэффициент dropout
    
    Returns:
    --------
    model : keras Model - построенная модель
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        print("TensorFlow/Keras не установлен. Пропускаем создание модели.")
        return None
    
    model = Sequential([
        LSTM(units, activation='relu', input_shape=input_shape, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dropout(dropout/2),
        Dense(output_shape)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=10):
    """
    Обучение LSTM модели
    
    Parameters:
    -----------
    X_train, y_train : arrays - обучающие данные
    X_val, y_val : arrays - валидационные данные
    epochs : int - число эпох обучения
    batch_size : int - размер батча
    patience : int - терпение для early stopping
    
    Returns:
    --------
    model : keras Model - обученная модель
    history : History - история обучения
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("TensorFlow/Keras не установлен.")
        return None, None
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print(f"\nПостроение LSTM модели...")
    print(f"Входная форма: {input_shape}")
    
    model = build_lstm_model(input_shape)
    
    if model is None:
        return None, None
    
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    print(f"\nОбучение модели ({epochs} эпох, batch_size={batch_size})...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_val, y_val, y_scaler):
    """
    Оценка качества модели
    
    Parameters:
    -----------
    model : keras Model - обученная модель
    X_val, y_val : arrays - валидационные данные
    y_scaler : sklearn Scaler - скалер для обратной трансформации
    
    Returns:
    --------
    metrics : dict - метрики качества
    """
    print("\n=== Оценка качества модели ===")
    
    # Предсказания
    y_pred_scaled = model.predict(X_val)
    
    # Обратное масштабирование
    y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_orig = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Метрики
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_val_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_orig, y_pred_orig)
    r2 = r2_score(y_val_orig, y_pred_orig)
    
    mape = np.mean(np.abs((y_val_orig - y_pred_orig) / (y_val_orig + 1e-10))) * 100
    
    print(f"\nМетрики на валидационной выборке:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
    
    return metrics, y_val_orig, y_pred_orig


def plot_training_history(history, save_dir=None):
    """
    Визуализация истории обучения
    
    Parameters:
    -----------
    history : keras History - история обучения
    save_dir : str - директория для сохранения графика
    """
    print("\n=== Визуализация истории обучения ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Динамика функции потерь')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Динамика средней абсолютной ошибки')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/training_history.png", dpi=150, bbox_inches='tight')
        print(f"График сохранён в {save_dir}/training_history.png")
    
    plt.show()


def plot_predictions(y_val, y_pred, save_dir=None):
    """
    Визуализация предсказаний vs фактических значений
    
    Parameters:
    -----------
    y_val : array - фактические значения
    y_pred : array - предсказанные значения
    save_dir : str - директория для сохранения графика
    """
    print("\n=== Визуализация предсказаний ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Временной ряд
    axes[0].plot(y_val, label='Фактические', linewidth=2)
    axes[0].plot(y_pred, label='Предсказанные', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Временная точка')
    axes[0].set_ylabel('Тепловая мощность (норм.)')
    axes[0].set_title('Сравнение фактических и предсказанных значений')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_val, y_pred, alpha=0.5, s=20)
    axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
    axes[1].set_xlabel('Фактические значения')
    axes[1].set_ylabel('Предсказанные значения')
    axes[1].set_title('Предсказанные vs Фактические значения')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/predictions_comparison.png", dpi=150, bbox_inches='tight')
        print(f"График сохранён в {save_dir}/predictions_comparison.png")
    
    plt.show()


def run_full_pipeline(data_path, params_path=None, save_dir='/workspace/SVRK',
                     history_window=10, forecast_horizon=1, variance_threshold=0.95):
    """
    Запуск полного цикла анализа
    
    Parameters:
    -----------
    data_path : str - путь к файлу с данными
    params_path : str - путь к файлу с параметрами
    save_dir : str - директория для сохранения результатов
    history_window : int - окно истории для LSTM (часов)
    forecast_horizon : int - горизонт прогноза (часов)
    variance_threshold : float - порог дисперсии для PCA
    """
    print("="*70)
    print("ИССЛЕДОВАНИЕ ВОЗМОЖНОСТИ ПРОГНОЗИРОВАНИЯ СОСТОЯНИЯ РЕАКТОРНОЙ УСТАНОВКИ")
    print("ВВЭР-1000 ПО ПОКАЗАНИЯМ СВРК С ПРИМЕНЕНИЕМ МЕТОДОВ МАШИННОГО ОБУЧЕНИЯ")
    print("="*70)
    
    # Создание директории для результатов
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Загрузка данных
    df, param_names = load_data(data_path, params_path)
    
    # Предобработка
    df_processed, preprocessing_info = preprocess_data(df, window_size=5)
    
    # Кластеризация
    labels, kmeans, cluster_info = cluster_analysis(df_processed, optimal_k=4)
    
    # Анализ профилей кластеров
    df_original = df.copy()
    cluster_profiles = analyze_cluster_profiles(df_original, labels, param_names)
    
    # PCA для каждого кластера
    pca_results = apply_pca_per_cluster(df_processed, labels, variance_threshold=variance_threshold)
    
    # Визуализация
    visualize_clusters_and_pca(df_processed, labels, pca_results, save_dir=save_dir)
    
    # Подготовка данных для LSTM
    X_train, y_train, X_val, y_val, scalers = prepare_lstm_data(
        df_processed, labels, pca_results,
        target_col='NAKЗ',
        history_window=history_window,
        forecast_horizon=forecast_horizon
    )
    
    # Обучение LSTM модели
    model, history = train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    if model is not None and history is not None:
        # Оценка модели
        metrics, y_val_orig, y_pred_orig = evaluate_model(model, X_val, y_val, scalers['y'])
        
        # Визуализация результатов
        plot_training_history(history, save_dir=save_dir)
        plot_predictions(y_val_orig, y_pred_orig, save_dir=save_dir)
        
        # Сохранение результатов
        results = {
            'preprocessing_info': preprocessing_info,
            'cluster_info': cluster_info,
            'pca_results_summary': {k: {'n_components': v['n_components'], 
                                        'explained_variance': v['explained_variance']} 
                                   for k, v in pca_results.items()},
            'model_metrics': metrics
        }
        
        import json
        with open(f"{save_dir}/results_summary.json", 'w', encoding='utf-8') as f:
            # Преобразование numpy типов для JSON сериализации
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            results_serializable = convert_numpy(results)
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\nРезультаты сохранены в {save_dir}/results_summary.json")
        
        print("\n" + "="*70)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО")
        print("="*70)
        
        return results
    
    return None


if __name__ == "__main__":
    # Пути к файлам
    DATA_PATH = '/workspace/Пример данных.xlsx'
    PARAMS_PATH = '/workspace/Список параметров.xlsx'
    SAVE_DIR = '/workspace/SVRK'
    
    # Запуск полного цикла анализа
    results = run_full_pipeline(
        data_path=DATA_PATH,
        params_path=PARAMS_PATH,
        save_dir=SAVE_DIR,
        history_window=10,  # 10 часов истории
        forecast_horizon=1,  # 1 час прогноза
        variance_threshold=0.95  # 95% дисперсии
    )
