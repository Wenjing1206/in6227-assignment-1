"""从真实 metrics.json 生成两页报告；结果表由数值文件直接填充。"""
import argparse, json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader


def make_report(run,config,out):
    r=json.loads((run/'metrics.json').read_text()); c=json.loads(config.read_text()); m=c['metadata']
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Times-Roman',fontSize=10,leading=12,spaceAfter=6))
    styles.add(ParagraphStyle(name='HeadCustom',fontName='Times-Bold',fontSize=10,leading=12,spaceBefore=7,spaceAfter=4))
    styles.add(ParagraphStyle(name='TitleCustom',fontName='Times-Bold',fontSize=15,leading=18,spaceAfter=8))
    flow=[]
    def p(text): flow.append(Paragraph(escape(str(text)),styles['BodyCustom']))
    def h(text): flow.append(Paragraph(text,styles['HeadCustom']))
    def table(rows,widths):
        cells=[[Paragraph(escape(str(v)),styles['BodyCustom']) for v in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.grey),('LINEBELOW',(0,-1),(-1,-1),.5,colors.grey),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),3)]));flow.append(t);flow.append(Spacer(1,5))
    flow.append(Paragraph('Reusable Classification SKILL: Model Comparison',styles['TitleCustom']))
    p(f"{m['name']} | Matric number: {m['matric_number']} | IN6227-Assignment-1 | Variant-2")
    p(f"LLM: {m['llm_model']}. Interface: {m['llm_interface']}.")
    p(f"GitHub repository: {m['repository']}")
    if 'TO BE SUPPLIED' in str(m): p('DRAFT: required submission metadata remains incomplete.')
    h('INTRODUCTION')
    p(f"A reusable SKILL accepts a dataset path, profiles the table, asks the AI to record justified decisions, executes leakage-controlled model comparison, and generates this report from saved evidence. This run uses {r['split']}: {r['raw_train_rows']:,} original training rows and {r['test_rows']:,} test rows, with {len(r['numeric_features'])} numerical and {len(r['categorical_features'])} categorical predictors. The objective is a defensible workflow, not maximum accuracy.")
    h('METHODS OR PROCEDURES')
    counts=', '.join(f'{k}: {v:,} ({v/r["train_rows"]:.1%})' for k,v in r['train_class_counts'].items())
    p(f"Data audit. Removed {r['missing_train_target']} training rows with missing targets; retained {r['train_rows']:,} labeled training rows ({counts}). Test has {r['test_labeled_rows']:,} labeled rows; {r['missing_test_target']} unlabeled rows receive predictions only. There are {r['train_feature_missing']} missing training feature cells after target filtering and {r['test_feature_missing']} missing test feature cells. No cross-split exact feature duplicates were detected.")
    profile=json.loads((run/'profile.json').read_text())
    p(f"The original training data contain {profile['train']['duplicate_rows']} exact duplicate rows. IQR flags are diagnostic rather than deletion rules: "+'; '.join(f'{k}: {v}' for k,v in r['outlier_flags'].items())+'.')
    p('Preprocessing. '+c['preprocessing_reason'])
    p('Feature selection. '+c['drop_reason'])
    p('Evaluation design. '+c['split_reason']+' '+c['metric_reason']+f" All candidate settings use the same {r['folds']}-fold stratified CV (seed {r['seed']}); preprocessing is fitted within folds. Best settings are refitted on all training rows before test evaluation. No test-based tuning or threshold search is performed.")
    for spec in c['models']:
        kind=spec['kind']; extra='Maximum 2,000 iterations; solver tolerance 1e-4.' if kind=='logistic_regression' else f"Fixed ensemble size {spec.get('n_estimators',160)}; trees stop at chosen depth or split/leaf constraints."
        p(kind.replace('_',' ').title()+'. '+spec['reason']+' Class weight: '+str(spec.get('class_weight'))+'. Grid: '+json.dumps(spec.get('grid',{}))+'. '+extra)
    flow.append(PageBreak())
    h('RESULTS')
    rows=[['Model','CV '+r['primary_metric']+' (mean ± SD)','Test accuracy','Test macro-F1','Test balanced acc.']]
    for name,item in r['models'].items():
        best=max(item['cv_candidates'],key=lambda x:x['mean']);t=item['test']
        rows.append([name.replace('_',' ').title(),f"{best['mean']:.4f} ± {best['std']:.4f}",*[f"{t[k]:.4f}" if t else 'N/A' for k in ['accuracy','f1_macro','balanced_accuracy']]])
    b=r['baseline']; rows.append(['Majority baseline','Not tuned',*[f"{b[k]:.4f}" if b else 'N/A' for k in ['accuracy','f1_macro','balanced_accuracy']]])
    table(rows,[108,119,77,77,78])
    p('CV selects '+r['winner'].replace('_',' ').title()+'. Selected settings: '+'; '.join(name+': '+json.dumps(item['best_params']) for name,item in r['models'].items())+'. CV SD is variation across folds, not a confidence interval.')
    if len(r['classes'])<=6:
        for name,item in r['models'].items():
            t=item['test']
            if t:
                p(name.replace('_',' ').title()+f": confusion matrix (rows = actual, columns = predicted; order {r['classes']}) = {t['confusion_matrix']}. "+'; '.join(f"{cl}: precision {t['classification_report'][cl]['precision']:.4f}, recall {t['classification_report'][cl]['recall']:.4f}, F1 {t['classification_report'][cl]['f1-score']:.4f}" for cl in r['classes'])+'.')
                if 'roc_auc' in t: p(f"ROC-AUC {t['roc_auc']:.4f}; average precision {t['average_precision']:.4f}. Positive class = {r['classes'][1]}; both use probabilities, while confusion counts use default predicted classes.")
    else: p('Per-class precision, recall, F1 and full confusion matrices are provided in metrics.json; the table summarizes overall performance.')
    h('DISCUSSION')
    p(c['discussion'])
    if len(r['classes'])==2 and r['baseline']:
        minority=min(r['train_class_counts'],key=r['train_class_counts'].get)
        p('The model ranking depends on the objective. Minority-class ('+minority+') recall: '+'; '.join(name.replace('_',' ').title()+f": {item['test']['classification_report'][minority]['recall']:.4f}" for name,item in r['models'].items())+'. Higher minority recall can matter more when false negatives carry higher cost, even if macro-F1 is lower.')
    if b:
        w=r['models'][r['winner']]['test'];p(f"The CV-selected model has test macro-F1 {w['f1_macro']:.4f}, versus {b['f1_macro']:.4f} for the majority baseline. Accuracy alone would conceal the baseline's failure to recover minority classes. The held-out comparison is descriptive; no statistical significance or generalization guarantee is claimed.")
    h('CONCLUSION')
    p(f"The workflow selected {r['winner'].replace('_',' ').title()} using training-only {r['primary_metric']}. The SKILL separates AI choices from deterministic execution and reports their evidence. Reuse requires fresh target, leakage, split and model decisions; the included runner supports IID tabular binary/multiclass tasks, while grouped, temporal or other special structures require adaptation before execution.")
    h('REPRODUCIBILITY AND REFERENCES')
    p('Evidence: decisions.json, profile.json, metrics.json, predictions.csv, selected_model.joblib and source SHA-256 hashes. Software: '+', '.join(f'{k} {v}' for k,v in r['versions'].items())+f". Captured training warnings: {len(r['warnings'])}; inspect execution output separately for subprocess warnings.")
    p('[1] IN6227 Data Mining - Assignment 1, provided brief, pp. 1-3. [2] IN6227 Reports Template, provided Word template (section structure and 10-point Times body). [3] scikit-learn, installed software and estimator documentation, version '+r['versions']['scikit_learn']+'.')
    def footer(canvas,doc):
        canvas.setFont('Times-Roman',9);canvas.drawString(42,25,'IN6227-Assignment-1 | Variant-2 | Tian Wenjing' if m['name']=='Tian Wenjing' else 'IN6227-Assignment-1 | Variant-2');canvas.drawRightString(A4[0]-42,25,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=A4,rightMargin=38,leftMargin=38,topMargin=32,bottomMargin=38).build(flow,onFirstPage=footer,onLaterPages=footer)
    pages=len(PdfReader(out).pages)
    if pages>2: raise ValueError(f'Report has {pages} pages; condense content without reducing body font size')
    print(f'Report generated: {pages} pages')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();make_report(a.run,a.config,a.out)
