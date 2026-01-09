from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd
synthesizer = GaussianCopulaSynthesizer.load('./distribution_model.pkl')

synthetic_data = synthesizer.sample(num_rows=10000)
synthetic_data.to_csv('./synthetic_data.csv',index=False, encoding='utf-8-sig')
