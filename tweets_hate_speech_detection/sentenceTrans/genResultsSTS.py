# -*- coding: utf-8 -*-

import psycopg2 as psql
import csv
from math import isnan

def main():
    con = psql.connect('dbname=claudette')
    cur = con.cursor()

    metrics = ['accuracy', 'precision', 'recall', 'gmean', 'f1', 'f2']
    metric_labels = ['\\textit{Accuracy}', '\\textit{Precision}', '\\textit{Recall}', '\\textit{Gmean}', '$F_{1}$', '$F_{2}$']

    models = [
        'rf_sentence_trans_stsb',
        'rf_sentence_trans_stsb_smote',
        'rf_sentence_trans_stsb_prowsyn',
        'rf_sentence_trans_stsb_pfsmote',
        'lr_sentence_trans_stsb',
        'lr_sentence_trans_stsb_smote',
        'lr_sentence_trans_stsb_prowsyn',
        'lr_sentence_trans_stsb_pfsmote',
        'knn_sentence_trans_stsb',
        'knn_sentence_trans_stsb_smote',
        'knn_sentence_trans_stsb_prowsyn',
        'knn_sentence_trans_stsb_pfsmote',
        'svmlin_sentence_trans_stsb',
        'svmlin_sentence_trans_stsb_smote',
        'svmlin_sentence_trans_stsb_prowsyn',
        'svmlin_sentence_trans_stsb_pfsmote',
        'svmrbf_sentence_trans_stsb',
        'svmrbf_sentence_trans_stsb_smote',
        'svmrbf_sentence_trans_stsb_prowsyn',
        'svmrbf_sentence_trans_stsb_pfsmote',
        'xgb_sentence_trans_stsb',
        'brf_sentence_trans_stsb',
        'ee_sentence_trans_stsb',
        'xgb_sentence_trans_stsb_weight',
        'knn_lmnn_sentence_trans_stsb',
        'svmlin_lmnn_sentence_trans_stsb',
        'mle_sentence_trans_stsb',
        'mle_knn_sentence_trans_stsb',
    ]

    model_labels = [
        'Random Forest & Sentence Transformer (STSb)',
        'Random Forest + SMOTE & Sentence Transformer (STSb)',
        'Random Forest + ProWSyn & Sentence Transformer (STSb)',
        'Random Forest + PFSMOTE & Sentence Transformer (STSb)',
        'Logistic Regression & Sentence Transformer (STS)',
        'Logistic Regression + SMOTE & Sentence Transformer (STS)',
        'Logistic Regression + ProWSyn & Sentence Transformer (STS)',
        'Logistic Regression + PFSMOTE & Sentence Transformer (STS)',
        'kNN & Sentence Transformer (STSb)',
        'kNN + SMOTE & Sentence Transformer (STSb)',
        'kNN + ProWSyn & Sentence Transformer (STSb)',
        'kNN + PFSMOTE & Sentence Transformer (STSb)',
        'SVM (Linear) & Sentence Transformer (STSb)',
        'SVM (Linear) + SMOTE & Sentence Transformer (STSb)',
        'SVM (Linear) + ProWSyn & Sentence Transformer (STSb)',
        'SVM (Linear) + PFSMOTE & Sentence Transformer (STSb)',
        'SVM (RBF) & Sentence Transformer (STSb)',
        'SVM (RBF) + SMOTE & Sentence Transformer (STSb)',
        'SVM (RBF) + ProWSyn & Sentence Transformer (STSb)',
        'SVM (RBF) + PFSMOTE & Sentence Transformer (STSb)',
        'XGBoost & Sentence Transformer (STSb)',
        'BRF & Sentence Transformer (STSb)',
        'EasyEnsemble & Sentence Transformer (STSb)',
        'Weighted XGBoost & Sentence Transformer (STSb)',
        'kNN + LMNN & Sentence Transformer (STSb)',
        'SVM (Linear) + LMNN & Sentence Transformer (STSb)',
        'MLE Ada & Sentence Transformer (STSb)',
        'MLE kNN & Sentence Transformer (STSb)',
    ]

    rows = []
    for metric in metrics:
        tmp_row = []
        for mdl in models:
            if mdl.startswith('mle'):
                cur.execute(f'''
            	    select avg+stddev/2
            	    from {mdl}_agg
            	    where metric = %s
            	''', (metric,))

            else:
                cur.execute(f'''
            	    select avg
            	    from {mdl}_agg
            	    where metric = %s
            	''', (metric,))

            tmp = cur.fetchone()
            if tmp is None or tmp[0] is None:
                tmp_row.append('NA')
            elif isnan(tmp[0]):
                tmp_row.append('NA')
            else:
                tmp_row.append(f'{tmp[0]:.3f}')

        lst = [0 if r == 'NA' else float(r) for r in tmp_row]
        max_val = max(lst)
        row = ['\\textbf{' + str(tmp_row[i]) + '}' if lst[i] == max_val 
            else str(tmp_row[i])
            for i in range(len(lst))]
        rows.append(row)

    with open(f'../out/comp.tex', 'w') as w:
        w.write('\\begin{tabularx}{\\textwidth}{ll!{\\vrule width .15em}')
        for i in range(len(metrics)):
            w.write('R')
        w.write('} \n \\specialrule{.15em}{.15em}{.15em} \n' )
        w.write(f'Model & Feature & {" & ".join(metric_labels)} \\\\ \n')
        w.write('\\specialrule{.15em}{.15em}{.15em} \n')

        for j in range(len(models)):
            #if models[j].endswith('uni') and j > 0:
            #    w.write('\\hdashline \n')
            w.write(f'{model_labels[j]}')
            for i in range(len(metrics)):
                w.write(f' & {rows[i][j]}')
            w.write(' \\\\ \n')

        #w.write('\\specialrule{.15em}{.15em}{.15em} \\end{tabularx}')

    cur.close()
    con.close()

if __name__ == "__main__":
    main()
