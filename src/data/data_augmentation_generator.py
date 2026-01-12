from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd

def synthetic_data_generator(path = './data/raw/', file_nm = 'synthetic_data.csv'):
  file_path = path+file
  synthesizer = GaussianCopulaSynthesizer.load('distribution_model.pkl')
  synthetic_data = synthesizer.sample(num_rows=10000)
  synthetic_data.to_csv(file_path,index=False, encoding='utf-8-sig')
  
