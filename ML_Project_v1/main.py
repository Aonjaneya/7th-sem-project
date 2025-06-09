import import_ipynb
import Trial_MPA
import Trial_PSO
import Trial_GWO
import Trial_SSA
import pandas as pd

df1 = pd.read_csv("diabetes.csv")
df1 = df1[(df1['Glucose'] != 0) & (df1['Insulin'] != 0)]
df1 = df1[(df1['BloodPressure'] != 0) & (df1['SkinThickness'] != 0) & (df1['BMI'] != 0) & (df1['DiabetesPedigreeFunction'] != 0)]
df1 = df1.reset_index(drop=True)
df2 = df1
X = df2.drop('Outcome', axis = 1)
y = df2.Outcome

model_MPA, accuracy_MPA, specificity_MPA, sensitivity_MPA, prevalence_MPA, f1_score_MPA = Trial_MPA.main(X, y)
model_PSO, accuracy_PSO, specificity_PSO, sensitivity_PSO, prevalence_PSO, f1_score_PSO = Trial_PSO.main(X, y)
model_GWO, accuracy_GWO, specificity_GWO, sensitivity_GWO, prevalence_GWO, f1_score_GWO = Trial_GWO.main(X, y)
model_SSA, accuracy_SSA, specificity_SSA, sensitivity_SSA, prevalence_SSA, f1_score_SSA = Trial_SSA.main(X, y)

def True_false(glu,bp,st,ins,bmi,dpf):
    if glu>=80 and glu<=125:
        g = 1
    else:
        g = 0
    if bp>=60 and bp<=80:
        b = 1
    else:
        b = 0
    if st>=10 and st<=26:
        s = 1
    else:
        s = 0
    if ins>=30 and ins<=300:
        i = 1
    else:
        i = 0
    if bmi>= 18.0 and bmi<=24.9:
        bm = 1
    else:
        bm = 0
    if dpf>=0.08 and dpf<=0.8:
        d = 1
    else:
        d = 0
    return g,b,s,i,bm,d

def convert(pre,glu,bp,st,ins,bmi,dpf,age):
    print(f"pre: {pre}, glu: {glu}, bp: {bp}, st: {st}, ins: {ins}, bmi: {bmi}, dpf: {dpf}, age: {age}")

    pr  = int(pre)
    gl  = int(glu)
    b = int(bp)
    s = int(st)
    i  = int(ins)
    bm  = float(bmi)
    dp  = float(dpf)
    ag  = int(age)
    return pr,gl,b,s,i,bm,dp,ag

def model(model,pre,glu,bp,st,ins,bmi,dpf,age):
    if model == 'MPA':
        p = model_MPA.predict([[pre,glu,bp,st,ins,bmi,dpf,age]])
        return p[0]
    if model == 'PSO':
        p = model_PSO.predict([[pre,glu,bp,st,ins,bmi,dpf,age]])
        return p[0]
    if model == 'GWO':
        p = model_GWO.predict([[pre,glu,bp,st,ins,bmi,dpf,age]])
        return p[0]
    if model == 'SSA':
        p = model_SSA.predict([[pre,glu,bp,st,ins,bmi,dpf,age]])
        return p[0]

from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_option = None
    if request.method == 'POST':
        selected_option = request.form.get('option')
        if selected_option == 'compare':
            return redirect(url_for('models'))
        else:
            return redirect(url_for('form'))
    return render_template('home.html')

@app.route('/models',methods=['GET', 'POST'])
def models():
    if request.method == 'POST':
        return redirect(url_for('form'))
    return render_template('compare.html',accuracy_MPA=accuracy_MPA, specificity_MPA=specificity_MPA,
                           sensitivity_MPA=sensitivity_MPA, prevalence_MPA=prevalence_MPA, f1_score_MPA=f1_score_MPA,
                           accuracy_PSO=accuracy_PSO, specificity_PSO=specificity_PSO,
                           sensitivity_PSO=sensitivity_PSO, prevalence_PSO=prevalence_PSO, f1_score_PSO=f1_score_PSO,
                           accuracy_GWO=accuracy_GWO, specificity_GWO=specificity_GWO,
                           sensitivity_GWO=sensitivity_GWO, prevalence_GWO=prevalence_GWO, f1_score_GWO=f1_score_GWO,
                           accuracy_SSA=accuracy_SSA, specificity_SSA=specificity_SSA,
                           sensitivity_SSA=sensitivity_SSA, prevalence_SSA=prevalence_SSA, f1_score_SSA=f1_score_SSA)
                           
    
@app.route('/form',methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        pre = request.form.get('pre', default='0')
        glu = request.form.get('glu')
        bp = request.form.get('bp')
        st = request.form.get('st')
        ins = request.form.get('ins')
        bmi = request.form.get('bmi')
        dpf = request.form.get('dpf')
        age = request.form.get('age')
        model_o = request.form.get('options')
        print(pre)
        print(type(pre)) 
        print(f"pre: {pre}, glu: {glu}, bp: {bp}, st: {st}, ins: {ins}, bmi: {bmi}, dpf: {dpf}, age: {age}")

        
        pre,glu,bp,st,ins,bmi,dpf,age = convert(pre,glu,bp,st,ins,bmi,dpf,age)

        a = model(model_o,pre,glu,bp,st,ins,bmi,dpf,age)

        if a==0:
            return redirect(url_for('Not_Diabetic',pre=pre,glu=glu,bp=bp,
                                    st=st,ins=ins,bmi=bmi,dpf=dpf,age=age,options=model_o))
        else:
            return redirect(url_for('Diabetic',pre=pre,glu=glu,bp=bp,
                                    st=st,ins=ins,bmi=bmi,dpf=dpf,age=age,options=model_o))
    else:
        return render_template('index.html')
    
    
@app.route('/Not_Diabetic',methods=['GET', 'POST'])
def Not_Diabetic():
    
    pre = request.args.get('pre', default='0')
    glu = request.args.get('glu')
    bp = request.args.get('bp')
    st = request.args.get('st')
    ins = request.args.get('ins')
    bmi = request.args.get('bmi')
    dpf = request.args.get('dpf')
    age = request.args.get('age')
    model_o = request.args.get('options')
    
    pre,glu,bp,st,ins,bmi,dpf,age = convert(pre,glu,bp,st,ins,bmi,dpf,age)

    g,b,s,i,bm,d = True_false(glu,bp,st,ins,bmi,dpf)
    
    if model_o == 'MPA':
        Name = 'Marine Predator Algorithm'
        Test_Acc = accuracy_MPA
    if model_o == 'PSO':
        Name = 'Particle swarm optimization'
        Test_Acc = accuracy_PSO
    if model_o == 'GWO':
        Name = 'Grey wolf Optimization'
        Test_Acc = accuracy_GWO
    if model_o == 'SSA':
        Name = 'Salp Swarm Algorithm'
        Test_Acc = accuracy_SSA
    if request.method == 'POST':
        pre = request.args.get('pre', default='0')
        glu = request.args.get('glu')
        bp = request.args.get('bp')
        st = request.args.get('st')
        ins = request.args.get('ins')
        bmi = request.args.get('bmi')
        dpf = request.args.get('dpf')
        age = request.args.get('age')
        model_o = request.form.get('options')
        
        pre,glu,bp,st,ins,bmi,dpf,age = convert(pre,glu,bp,st,ins,bmi,dpf,age)

        g,b,s,i,bm,d = True_false(glu,bp,st,ins,bmi,dpf)
        a = model(model_o,pre,glu,bp,st,ins,bmi,dpf,age)
        
        if a==0:
            if model_o == 'MPA':
                Name = 'Marine Predator Algorithm'
                Test_Acc = accuracy_MPA
            if model_o == 'PSO':
                Name = 'Particle swarm optimization'
                Test_Acc = accuracy_PSO
            if model_o == 'GWO':
                Name = 'Grey wolf Optimization'
                Test_Acc = accuracy_GWO
            if model_o == 'SSA':
                Name = 'Salp Swarm Algorithm'
                Test_Acc = accuracy_SSA
            return render_template('Yes.html',model_name=Name, Test_Acc=Test_Acc, glu_v=glu, glu=g, bp_v=bp, bp=b, st_v=st, st=s, ins_v=ins, ins=i, bmi_v=bmi, bmi=bm, dpf_v=dpf, dpf=d)
        else:
            return redirect(url_for('Diabetic',pre=pre,glu=glu,bp=bp,
                                    st=st,ins=ins,bmi=bmi,dpf=dpf,age=age,options=model_o))
        
    return render_template('Yes.html',model_name=Name, Test_Acc=Test_Acc, glu_v=glu, glu=g, bp_v=bp, bp=b, st_v=st, st=s, ins_v=ins, ins=i, bmi_v=bmi, bmi=bm, dpf_v=dpf, dpf=d)


@app.route('/Diabetic',methods=['GET', 'POST'])
def Diabetic():
    pre = request.args.get('pre', default='0')
    glu = request.args.get('glu')
    bp = request.args.get('bp')
    st = request.args.get('st')
    ins = request.args.get('ins')
    bmi = request.args.get('bmi')
    dpf = request.args.get('dpf')
    age = request.args.get('age')
    model_o = request.args.get('options')
    
    pre,glu,bp,st,ins,bmi,dpf,age = convert(pre,glu,bp,st,ins,bmi,dpf,age)

    g,b,s,i,bm,d = True_false(glu,bp,st,ins,bmi,dpf)
    
    if model_o == 'MPA':
        Name = 'Marine Predator Algorithm'
        Test_Acc = accuracy_MPA
    if model_o == 'PSO':
        Name = 'Particle swarm optimization'
        Test_Acc = accuracy_PSO
    if model_o == 'GWO':
        Name = 'Grey wolf Optimization'
        Test_Acc = accuracy_GWO
    if model_o == 'SSA':
        Name = 'Salp Swarm Algorithm'
        Test_Acc = accuracy_SSA
    if request.method == 'POST':
        pre = request.args.get('pre', default='0')
        glu = request.args.get('glu')
        bp = request.args.get('bp')
        st = request.args.get('st')
        ins = request.args.get('ins')
        bmi = request.args.get('bmi')
        dpf = request.args.get('dpf')
        age = request.args.get('age')
        model_o = request.form.get('options')
        pre,glu,bp,st,ins,bmi,dpf,age = convert(pre,glu,bp,st,ins,bmi,dpf,age)

        g,b,s,i,bm,d = True_false(glu,bp,st,ins,bmi,dpf)
        a = model(model_o,pre,glu,bp,st,ins,bmi,dpf,age)
        
        if a==0:
            return redirect(url_for('Not_Diabetic',pre=pre,glu=glu,bp=bp,
                                    st=st,ins=ins,bmi=bmi,dpf=dpf,age=age,options=model_o))
        else:
            if model_o == 'MPA':
                Name = 'Marine Predator Algorithm'
                Test_Acc = accuracy_MPA
            if model_o == 'PSO':
                Name = 'Particle swarm optimization'
                Test_Acc = accuracy_PSO
            if model_o == 'GWO':
                Name = 'Grey wolf Optimization'
                Test_Acc = accuracy_GWO
            if model_o == 'SSA':
                Name = 'Salp Swarm Algorithm'
                Test_Acc = accuracy_SSA
            return render_template('No.html',model_name=Name, Test_Acc=Test_Acc, glu_v=glu, glu=g, bp_v=bp, bp=b, st_v=st, st=s, ins_v=ins, ins=i, bmi_v=bmi, bmi=bm, dpf_v=dpf, dpf=d)
        
    return render_template('No.html',model_name=Name, Test_Acc=Test_Acc, glu_v=glu, glu=g, bp_v=bp, bp=b, st_v=st, st=s, ins_v=ins, ins=i, bmi_v=bmi, bmi=bm, dpf_v=dpf, dpf=d)
    
if __name__ == '__main__':
    app.run(debug=True)