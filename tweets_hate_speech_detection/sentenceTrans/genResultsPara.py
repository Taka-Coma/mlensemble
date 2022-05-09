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
        'rf_sentence_trans_para',
        'rf_sentence_trans_para_smote',
        'rf_sentence_trans_para_prowsyn',
        'rf_sentence_trans_para_pfsmote',
        'lr_sentence_trans_para',
        'lr_sentence_trans_para_smote',
        'lr_sentence_trans_para_prowsyn',
        'lr_sentence_trans_para_pfsmote',
        'knn_sentence_trans_para',
        'knn_sentence_trans_para_smote',
        'knn_sentence_trans_para_prowsyn',
        'knn_sentence_trans_para_pfsmote',
        'svmlin_sentence_trans_para',
        'svmlin_sentence_trans_para_smote',
        'svmlin_sentence_trans_para_prowsyn',
        'svmlin_sentence_trans_para_pfsmote',
        'svmrbf_sentence_trans_para',
        'xgb_sentence_trans_para',
        'brf_sentence_trans_para',
        'ee_sentence_trans_para',
        'xgb_sentence_trans_para_weight',
        'knn_lmnn_sentence_trans_para',
        'mle_sentence_trans_para',
        'mle_knn_sentence_trans_para',
    ]

    model_labels = [
        'Random Forest & Sentence Transformer (para)',
        'Random Forest + SMOTE & Sentence Transformer (para)',
        'Random Forest + ProWSyn & Sentence Transformer (para)',
        'Random Forest + PFSMOTE & Sentence Transformer (para)',
        'Logistic Regression & Sentence Transformer (para)',
        'Logistic Regression + SMOTE & Sentence Transformer (para)',
        'Logistic Regression + ProWSyn & Sentence Transformer (para)',
        'Logistic Regression + PFSMOTE & Sentence Transformer (para)',
        'kNN & Sentence Transformer (para)',
        'kNN + SMOTE & Sentence Transformer (para)',
        'kNN + ProWSyn & Sentence Transformer (para)',
        'kNN + PFSMOTE & Sentence Transformer (para)',
        'SVM (Linear) & Sentence Transformer (para)',
        'SVM (Linear) + SMOTE & Sentence Transformer (para)',
        'SVM (Linear) + ProWSyn & Sentence Transformer (para)',
        'SVM (Linear) + PFSMOTE & Sentence Transformer (para)',
        'SVM (RBF) & Sentence Transformer (para)',
        'XGBoost & Sentence Transformer (para)',
        'BRF & Sentence Transformer (para)',
        'EasyEnsemble & Sentence Transformer (para)',
        'Weighted XGBoost & Sentence Transformer (para)',
        'kNN + LMNN & Sentence Transformer (para)',
        'MLE Ada & Sentence Transformer (para)',
        'MLE kNN & Sentence Transformer (para)',
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
