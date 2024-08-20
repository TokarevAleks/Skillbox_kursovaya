import datetime
import dill
import os
import pandas as pd
import numpy as np
import sys
import io

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from functools import reduce
from pyspark.sql import SparkSession

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# Ссылки на датасеты в формате parquet и csv
files_id_data = 'C:/Users/Tokar/Skillbox MJ Junior/Итоговая работа/train_data'
file_id_target = 'C:/Users/Tokar/Skillbox MJ Junior/Итоговая работа/train_target.csv'


def prepare_dataset():
    """
    Функция для подготовки и объединения датасетов,
    выполнения агрегаций и создания итогового Pandas DataFrame.
    """
    # Инициализация SparkSession с настройками для работы с большими данными
    spark = SparkSession.builder \
        .appName("Combine and Process Data") \
        .config("spark.executor.memory", "12g") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.network.timeout", "600s") \
        .config("spark.sql.debug.maxToStringFields", "1000") \
        .config("spark.driver.maxResultSize", "8g") \
        .getOrCreate()

    # Установить уровень логирования на ERROR, чтобы скрыть предупреждения
    spark.sparkContext.setLogLevel("ERROR")

    # Чтение и объединение данных из всех файлов .parquet в указанной директории
    file_paths = [os.path.join(files_id_data, f) for f in os.listdir(files_id_data) if f.endswith('.pq')]
    dataframes = [spark.read.parquet(file_path) for file_path in file_paths]

    # Функция для объединения всех DataFrame'ов в один
    def union_all(dfs):
        return reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), dfs)

    # Объединяем все DataFrame'ы, сортируем по 'id' и удаляем дубликаты
    combined_df = union_all(dataframes) if dataframes else None
    combined_df = combined_df.orderBy('id').dropDuplicates()

    # Функция для выполнения агрегаций на объединенном DataFrame
    def aggregate_data(df_clear):
        # Создаем временное представление для выполнения SQL-запросов
        df_clear.createOrReplaceTempView("combined_data")

        # Определяем колонки и соответствующие агрегатные функции
        columns_agg = {
            'rn': ['COUNT'],
            'pre_since_opened': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_since_confirmed': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_pterm': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_fterm': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_till_pclose': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_till_fclose': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_loans_credit_limit': ['AVG', 'MEDIAN'],
            'pre_loans_next_pay_summ': ['MIN', 'MAX'],
            'pre_loans_outstanding': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_loans_max_overdue_sum': ['AVG', 'MIN', 'MAX'],
            'pre_loans_credit_cost_rate': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_loans5': ['AVG', 'MIN', 'MAX'],
            'pre_loans530': ['AVG', 'MIN', 'MAX'],
            'pre_loans3060': ['AVG', 'MAX'],
            'pre_loans6090': ['AVG'],
            'pre_loans90': ['MAX'],
            'is_zero_loans5': ['AVG', 'MAX'],
            'is_zero_loans530': ['AVG', 'MEDIAN'],
            'is_zero_loans3060': ['AVG'],
            'is_zero_loans6090': ['AVG'],
            'is_zero_loans90': ['AVG', 'MEDIAN'],
            'pre_util': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'pre_over2limit': ['AVG', 'MAX', 'MEDIAN'],
            'pre_maxover2limit': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'is_zero_util': ['AVG'],
            'is_zero_over2limit': ['AVG', 'MAX'],
            'is_zero_maxover2limit': ['AVG'],
            'pclose_flag': ['AVG', 'MAX', 'MEDIAN'],
            'fclose_flag': ['AVG'],
            **{f'enc_paym_{i}': ['AVG', 'MIN', 'MAX', 'MEDIAN'] for i in range(25)},
            'enc_loans_account_holder_type': ['AVG', 'MAX'],
            'enc_loans_credit_status': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'enc_loans_credit_type': ['AVG', 'MIN', 'MAX', 'MEDIAN'],
            'enc_loans_account_cur': ['AVG']
        }

        # Формируем SQL-запрос для выполнения агрегатных операций
        sql_parts = []
        for col, aggs in columns_agg.items():
            for agg in aggs:
                if agg == 'MEDIAN':
                    part = f"percentile_approx({col}, 0.5) as {col}_median"
                elif agg == 'MODE':
                    part = f"(SELECT {col} FROM combined_data GROUP BY {col} ORDER BY COUNT({col}) DESC LIMIT 1) as {col}_mode"
                else:
                    part = f"{agg}({col}) as {col}_{agg.lower()}"
                sql_parts.append(part)

        # Создаем финальный SQL-запрос, который выполнит все агрегаты и сгруппирует данные по 'id'
        sql_query = f"SELECT id, {', '.join(sql_parts)} FROM combined_data GROUP BY id"
        return spark.sql(sql_query)

    # Функция для объединения агрегированных данных с таргетными данными по 'id'
    def join_with_target(result_df, target_path):
        target_df = spark.read.csv(target_path, header=True, inferSchema=True)
        return result_df.join(target_df, "id", "inner")

    # Основной процесс
    result_df = aggregate_data(combined_df)
    final_df = join_with_target(result_df, file_id_target)

    # Перевод Spark DataFrame в Pandas DataFrame
    pandas_df = final_df.toPandas()

    # Переименование колонки 'flag' в 'target'
    pandas_df = pandas_df.rename(columns={'flag': 'target'})

    # Закрытие сессии после завершения работы
    spark.stop()

    return pandas_df

def main():
    print('Loan Prediction Pipeline')

    df = prepare_dataset()

    X = df.drop(['target', 'id'], axis=1)
    y = df['target']

    numerical_features = make_column_selector(dtype_include='number')

    numerical_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])

    preprocessor_feat = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features)
    ])

    models = (
        LogisticRegression(solver='saga', max_iter=2000),
        RandomForestClassifier(),
        MLPClassifier(),
        XGBClassifier(
            colsample_bytree=0.80854,
            learning_rate=0.035541,
            max_depth=5,
            n_estimators=1362,
            subsample=0.877786,
            eval_metric='auc',
            tree_method="hist"
        )
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor_feat', preprocessor_feat),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')

        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    with open('default_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Default model',
                'author': 'Aleksandr Tokarev',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file, recurse=True)


if __name__ == '__main__':
    main()


