import pandas as pd

def get_recommendation(rec_dict):
    df = pd.read_csv("vehicles.csv")
    df['score'] = 0
    
    for column, values in rec_dict.items():
        df['score'] += df[column].isin(values).astype(int)
    
    df_sorted = df.sort_values(by='score', ascending=False)
    max_score = df_sorted['score'].max()
    top_rows = df_sorted[df_sorted['score'] == max_score]
    
    if len(top_rows) >= 2:
        best_rows = top_rows.head(2)
    else:
        best_rows = top_rows.sample(n=2, replace=True) if len(top_rows) == 1 else df.sample(n=2)
    
    best_rows = best_rows.drop(columns=['Stock', 'Model', 'EngineDisplacement', 'MSRP', 'BookValue', 'Invoice', 'Certified', 'Options', 'Style_Description', 'Ext_Color_Code', 'Int_Color_Generic', 'Int_Color_Code', 'Int_Upholstery', 'Engine_Block_Type', 'Engine_Aspiration_Type', 'Transmission_Description', 'Fuel_Type', 'EPAClassification', 'Wheelbase_Code', 'Internet_Price', 'ExtColorHexCode', 'IntColorHexCode', 'EngineDisplacementCubicInches', 'score'])
    arr = [best_rows.columns.tolist(), best_rows.iloc[0].tolist(), best_rows.iloc[1].tolist()]

    return arr