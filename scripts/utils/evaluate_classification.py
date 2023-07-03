import argparse
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
)
from statsmodels.formula.api import logit
from statsmodels.tools.eval_measures import bic, aic

def logistic_regression(data, formula, output_file):
    model = logit(formula=formula, data=data)
    target_variable = formula.split("~")[0].strip()
    print(f"Training model for {target_variable} ~ {formula.split('~')[1].strip()}\n")
    print(f"Total: {data.shape[0]}")
    print(f"Positive: {data[data[target_variable]==1].shape[0]}")
    print(f"Fraction positive: {data[data[target_variable]==1].shape[0]/data.shape[0]}")
    model_fit = model.fit()
    with open(output_file, "w") as f:
        f.write(model_fit.summary().as_text()+"\n")
        f.write(f"BIC: {bic(model_fit.llf, model_fit.nobs, model_fit.df_model)}\n")
        f.write(f"AIC: {aic(model_fit.llf, model_fit.nobs, model_fit.df_model)}\n")
        f.write(f"ROC AUC: {roc_auc_score(data[target_variable], model_fit.predict())}\n")
        f.write(f"PR AUC: {average_precision_score(data[target_variable], model_fit.predict())}\n")
    model_fit.save(output_file.replace(".txt", ".pkl"))


    return model_fit

def main(input_file, formula_list_file, output_file):
    data = pd.read_csv(input_file)
    with open(formula_list_file) as f:
        formulas = f.readlines()
    for formula in formulas:
        model_fit = logistic_regression(data, formula, output_file)
    return model_fit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_file", type=str, required=True)
    parser.add_argument("-f",
                        "--formula_list", type=str, required=True)
    parser.add_argument("-o",
                        "--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args.input_file, args.formula_list, args.output_file)