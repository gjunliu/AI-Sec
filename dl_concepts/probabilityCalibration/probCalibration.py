import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

'''
Probability Calibration concept learning and practicing.

Resource reference:
https://github.com/numeristical/resources/blob/master/CalibrationWorkshop/Calibration_Workshop_1.ipynb
'''

testData = 'lab_vital_icu_table.csv'

def main():
    # load dataset
    lab_aug_df = pd.read_csv(testData)

    for i in range(len(lab_aug_df.columns)):
        if lab_aug_df.iloc[:,i].dtype!='O':
            lab_aug_df.iloc[:,i].fillna(lab_aug_df.iloc[:,i].median(),inplace=True)
    # Choose a subset of variables
    feature_set_1 = ['bun_min',
           'bun_max', 'wbc_min', 'wbc_max','sysbp_max', 'sysbp_min']

    X_1 = lab_aug_df.loc[:,feature_set_1]
    y = lab_aug_df['hospital_expire_flag']

    train_perc = .6
    calib_perc = .05
    test_perc = 1-train_perc-calib_perc
    rs = 42

    X_train_calib_1, X_test_1, y_train_calib_1, y_test_1 = train_test_split(X_1, y, test_size=test_perc, random_state=rs)
    X_train_1, X_calib_1, y_train_1, y_calib_1 = train_test_split(X_train_calib_1, y_train_calib_1,
                                                                  test_size=calib_perc/(1-test_perc),
                                                                  random_state=rs)
    rfmodel1 = RandomForestClassifier(n_estimators = 500, class_weight='balanced_subsample',
                                  random_state=rs, n_jobs=-1 )
    rfmodel1.fit(X_train_1,y_train_1)
    # get the calibration training data
    calibset_preds_uncalib_1 = rfmodel1.predict_proba(X_calib_1)[:,1]
    testset_preds_uncalib_1 = rfmodel1.predict_proba(X_test_1)[:,1]
    print(calibset_preds_uncalib_1[:5])

    # Platt scaling (logistic calibration)
    lr = LogisticRegression(C=99999999999, solver='lbfgs')
    lr.fit(calibset_preds_uncalib_1.reshape(-1,1), y_calib_1)
    calibset_platt_probs = lr.predict_proba(calibset_preds_uncalib_1.reshape(-1,1))[:,1]
    testset_platt_probs = lr.predict_proba(testset_preds_uncalib_1.reshape(-1,1))[:,1]
    print(calibset_platt_probs[:5])

    # isotonic regression
    iso = IsotonicRegression(out_of_bounds = 'clip')
    iso.fit(calibset_preds_uncalib_1, y_calib_1)
    calibset_iso_probs = iso.predict(calibset_preds_uncalib_1)
    testset_iso_probs = iso.predict(testset_preds_uncalib_1)
    print(calibset_iso_probs[:5])


if __name__ == '__main__':
    main()
