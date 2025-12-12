import pickle
import inflection
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        # scalers carregados do disco
        self.competition_distance_scaler = pickle.load(open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler = pickle.load(open(self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler = pickle.load(open(self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))

    def data_cleaning(self, df1):
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']
        snakecase = lambda x: inflection.underscore(x)
        df1.columns = list(map(snakecase, cols_old))

        df1['date'] = pd.to_datetime(df1['date'])

        # substituindo NaNs
        df1['competition_distance'] = np.where(pd.isna(df1['competition_distance']), 200000.0, df1['competition_distance'])
        df1['competition_open_since_month'] = np.where(pd.isna(df1['competition_open_since_month']), df1['date'].dt.month, df1['competition_open_since_month'])
        df1['competition_open_since_year'] = np.where(pd.isna(df1['competition_open_since_year']), df1['date'].dt.year, df1['competition_open_since_year'])
        df1['promo2_since_week'] = np.where(pd.isna(df1['promo2_since_week']), df1['date'].dt.isocalendar().week, df1['promo2_since_week'])
        df1['promo2_since_year'] = np.where(pd.isna(df1['promo2_since_year']), df1['date'].dt.year, df1['promo2_since_year'])

        # promo_interval
        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                     7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        df1['promo_interval'] = df1['promo_interval'].fillna(0)
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo'] = df1[['promo_interval','month_map']].apply(
            lambda x: 0 if x['promo_interval']==0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0,
            axis=1
        )

        # converter colunas para int
        cols_int = ['competition_open_since_month', 'competition_open_since_year', 'promo2_since_week', 'promo2_since_year']
        df1[cols_int] = df1[cols_int].astype(int)

        return df1

    def feature_engineering(self, df2):
        df2['year'] = df2['date'].dt.year
        df2['month'] = df2['date'].dt.month
        df2['day'] = df2['date'].dt.day
        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition since
        df2['competition_since'] = df2.apply(
            lambda x: datetime(year=int(x['competition_open_since_year']),
                               month=int(x['competition_open_since_month']),
                               day=1),
            axis=1
        )
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since']) / timedelta(days=30)).astype(int)

        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(
            lambda x: datetime.strptime(x + '-1','%Y-%W-%w') - timedelta(days=7)
        )
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since']) / timedelta(weeks=1)).astype(int)

        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x=='a' else 'extra' if x=='b' else 'extended')

        # state_holiday
        df2['state_holiday'] = df2['state_holiday'].apply(
            lambda x: 'public_holiday' if x=='a' else 'easter_holiday' if x=='b' else 'christmas' if x=='c' else 'regular_day'
        )

        df2 = df2[df2['open'] != 0]
        df2 = df2.drop(['open','promo_interval','month_map'], axis=1)

        return df2

    def data_preparation(self, df5):
        # aplicar transform
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']])
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5[['competition_time_month']])
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(df5[['promo_time_week']])
        df5['year'] = self.year_scaler.transform(df5[['year']])

        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df5['store_type'] = self.store_type_scaler.transform(df5['store_type'].astype(str))

        # assortment - Ordinal Encoding
        assortment_dict = {'basic':1, 'extra':2, 'extended':3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        # transformações trigonométricas
        df5['day_of_week_sin'] = np.sin(df5['day_of_week'] * 2. * np.pi/7)
        df5['day_of_week_cos'] = np.cos(df5['day_of_week'] * 2. * np.pi/7)
        df5['month_sin'] = np.sin(df5['month'] * 2. * np.pi/12)
        df5['month_cos'] = np.cos(df5['month'] * 2. * np.pi/12)
        df5['day_sin'] = np.sin(df5['day'] * 2. * np.pi/30)
        df5['day_cos'] = np.cos(df5['day'] * 2. * np.pi/30)
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi / 52))).astype('float64')
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi / 52))).astype('float64')
        
        # Garantir que tudo seja numérico
        df5 = df5.apply(pd.to_numeric, errors='coerce')


        cols_selected = [
            'store','promo','school_holiday','store_type','assortment',
            'competition_distance','competition_open_since_month','competition_open_since_year',
            'promo2','promo2_since_week','promo2_since_year','competition_time_month','promo_time_week',
            'day_of_week_sin','day_of_week_cos','month_sin','month_cos','day_sin','day_cos',
            'week_of_year_sin','week_of_year_cos'
        ]

        return df5[cols_selected]

    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        return original_data.to_json( orient='records', date_format='iso' )
