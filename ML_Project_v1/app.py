import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

df1 = pd.read_csv("diabetes.csv")
df1 = df1[(df1['Glucose'] != 0) & (df1['Insulin'] != 0)]
df1 = df1[(df1['BloodPressure'] != 0) & (df1['SkinThickness'] != 0) & (df1['BMI'] != 0) & (df1['DiabetesPedigreeFunction'] != 0)]
df1 = df1.reset_index(drop=True)
df2 = df1
X = df2.drop('Outcome', axis = 1)
y = df2.Outcome

# Marine Predator Algorithm



def update_agents(agents, fitness, best_agent, alpha=0.5, beta=0.05, bounds=None):
    num_agents, dimension = agents.shape

    # Ensure bounds are provided
    if bounds is None:
        bounds = (np.array([10, 4, 2, 2]), np.array([150, 20, 10, 10]))

    lower_bound, upper_bound = bounds

    for i in range(num_agents):
        distance = best_agent - agents[i]

        if np.random.rand() < alpha:
            agents[i] += alpha * distance * np.random.rand(dimension)
        else:
            agents[i] += beta * (np.random.rand(dimension) - 0.5)

        agents[i] = np.clip(agents[i], lower_bound, upper_bound)

    return agents

# Hyperparameter optimization using MPA
def hyperparameter_optimization(X, y, max_iterations=100, num_agents=15):
    # Initialize agents randomly within specified bounds
    agents = np.random.rand(num_agents, 4)
    agents[:, 0] = agents[:, 0] * 140 + 10  # n_estimators (10 to 90)
    agents[:, 1] = agents[:, 1] * 12 + 8     # max_depth (4 to 20)
    agents[:, 2] = agents[:, 2] * 8 + 2    # min_samples_split (2 to 10)
    agents[:, 3] = agents[:, 3] * 8 + 2    # min_samples_leaf (2 to 10)

    best_fitness = float('-inf')
    best_agent = None

    for iteration in range(max_iterations):
        for i in range(num_agents):
            # Extract hyperparameters
            n_estimators = int(agents[i, 0])
            max_depth = int(agents[i, 1])
            min_samples_split = int(agents[i, 2])
            min_samples_leaf = int(agents[i, 3])
            if min_samples_split <= 2:
                min_samples_split = 2

            # Create and evaluate the RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         random_state=42)

            # Fit the model and evaluate fitness (accuracy)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fitness = accuracy_score(y_test, y_pred)

            # Update the best agent if current fitness is better
            if fitness > best_fitness:
                best_fitness = fitness
                best_agent = agents[i]

        # Update agents based on MPA rules
        agents = update_agents(agents, fitness, best_agent)

    return best_agent, best_fitness

best_hyperparams, best_accuracy = hyperparameter_optimization(X_train, y_train)
n_estimators,max_depth,min_samples_split,min_samples_leaf = best_hyperparams
n_estimators,max_depth,min_samples_split,min_samples_leaf = round(n_estimators),round(max_depth),round(min_samples_split),round(min_samples_leaf)
print("n_estimators = ", n_estimators, ", max_depth = ", max_depth)
print("min_samples_split = ", min_samples_split, ", min_samples_leaf = ", min_samples_leaf)
print("Optimal accuracy = ",best_accuracy)

model_1 = RandomForestClassifier(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         random_state=42)
model_1.fit(X_train.values, y_train.values)
model_1_train = model_1.predict(X_train.values)
Train_Accuracy =format(metrics.accuracy_score(y_train.values, model_1_train))
number_float = float(Train_Accuracy)
number_float = number_float * 100
Train_Acc = round(number_float, 1)
pred_1 = model_1.predict(X_test.values)
Accuracy = format(metrics.accuracy_score(y_test.values, pred_1))
number_float = float(Accuracy)
number_float = number_float * 100
Test_Acc = round(number_float, 1)

# Particle Swarm Optimizer



class Particle:
    def __init__(self, n_estimators_range, max_depth_range):
        self.position = np.array([
            np.random.randint(n_estimators_range[0], n_estimators_range[1]),  # n_estimators
            np.random.randint(max_depth_range[0], max_depth_range[1])         # max_depth
        ])
        self.velocity = np.random.uniform(-1, 1, 2)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')  # Start with infinity

# Define the objective function
def evaluate_particle(particle, X_train, y_train, X_test, y_test):
    n_estimators = int(particle.position[0])
    max_depth = int(particle.position[1])
    
    model_PSO = RandomForestClassifier(n_estimators=n_estimators, 
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       random_state=42)
    model_PSO.fit(X_train, y_train)
    predictions = model_PSO.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# PSO algorithm
def pso(num_particles, n_estimators_range, max_depth_range, max_iter):
    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize particles
    particles = [Particle(n_estimators_range, max_depth_range) for _ in range(num_particles)]
    global_best_position = np.copy(particles[0].best_position)
    global_best_value = float('inf')

    # Hyperparameters
    w = 0.5  # inertia weight
    c1 = 1.5  # cognitive (personal) weight
    c2 = 1.5  # social (global) weight

    # Main loop
    for iter in range(max_iter):
        for particle in particles:
            # Evaluate particle
            current_value = evaluate_particle(particle, X_train, y_train, X_test, y_test)

            # Update personal best
            if current_value < particle.best_value:
                particle.best_value = current_value
                particle.best_position = np.copy(particle.position)

            # Update global best
            if current_value < global_best_value:
                global_best_value = current_value
                global_best_position = np.copy(particle.position)

            # Update velocity
            r1, r2 = np.random.rand(2)
            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (global_best_position - particle.position))

            # Update position
            particle.position += particle.velocity
            # Ensure position is within bounds
            particle.position[0] = np.clip(particle.position[0], n_estimators_range[0], n_estimators_range[1])
            particle.position[1] = np.clip(particle.position[1], max_depth_range[0], max_depth_range[1])

        # Print progress
        print(f"Iteration {iter + 1}/{max_iter}, Global Best Value (Accuracy): {global_best_value}")

    return global_best_position, global_best_value

# Parameters
num_particles = 30
n_estimators_range = (10, 200)  # Range for n_estimators
max_depth_range = (1, 10)        # Range for max_depth
max_iter = 50

# Run PSO
best_position, best_value = pso(num_particles, n_estimators_range, max_depth_range, max_iter)

print(f"Best Position (n_estimators, max_depth): {best_position}")
print(f"Best Value (Accuracy): {best_value}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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
    
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        pre = request.form.get('pre')
        glu = request.form.get('glu')
        bp = request.form.get('bp')
        st = request.form.get('st')
        ins = request.form.get('ins')
        bmi = request.form.get('bmi')
        dpf = request.form.get('dpf')
        age = request.form.get('age')
        print(type(pre))

        pre = int(pre)
        glu = int(glu)
        bp = int(bp)
        st = int(st)
        ins = int(ins)
        bmi = float(bmi)
        dpf = float(dpf)
        age = int(age)
        

        b = model_1.predict([[pre,glu,bp,st,ins,bmi,dpf,age]])
        a=b[0]
        if a==0:
            return redirect(url_for('Not_Diabetic',pre=pre,glu=glu,bp=bp,st=st,ins=ins,bmi=bmi,dpf=dpf,age=age))
        else:
            return redirect(url_for('Diabetic',pre=pre,glu=glu,bp=bp,st=st,ins=ins,bmi=bmi,dpf=dpf,age=age))
    else:
        return render_template('index.html', Train_Acc=Train_Acc, Test_Acc=Test_Acc)

@app.route('/Not_Diabetic')
def Not_Diabetic():
    pre = request.form.get('pre')
    glu = request.args.get('glu')
    bp = request.args.get('bp')
    st = request.args.get('st')
    ins = request.args.get('ins')
    bmi = request.args.get('bmi')
    dpf = request.args.get('dpf')
    age = request.form.get('age')

    glu = int(glu)
    bp = int(bp)
    st = int(st)
    ins = int(ins)
    bmi = float(bmi)
    dpf = float(dpf)

    g,b,s,i,bm,d = True_false(glu,bp,st,ins,bmi,dpf)
    
    return render_template('Yes.html', Train_Acc=Train_Acc, Test_Acc=Test_Acc, glu_v=glu, glu=g, bp_v=bp, bp=b, st_v=st, st=s, ins_v=ins, ins=i, bmi_v=bmi, bmi=bm, dpf_v=dpf, dpf=d)
@app.route('/Diabetic')
def Diabetic():
    pre = request.form.get('pre')
    glu = request.args.get('glu')
    bp = request.args.get('bp')
    st = request.args.get('st')
    ins = request.args.get('ins')
    bmi = request.args.get('bmi')
    dpf = request.args.get('dpf')
    age = request.form.get('age')
    
    glu = int(glu)
    bp = int(bp)
    st = int(st)
    ins = int(ins)
    bmi = float(bmi)
    dpf = float(dpf)

    g,b,s,i,bm,d = True_false(glu,bp,st,ins,bmi,dpf)
    
    return render_template('No.html', Train_Acc=Train_Acc, Test_Acc=Test_Acc, glu_v=glu, glu=g, bp_v=bp, bp=b, st_v=st, st=s, ins_v=ins, ins=i, bmi_v=bmi, bmi=bm, dpf_v=dpf, dpf=d)

if __name__ == '__main__':
    app.run(debug=True)