from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

class AlcoholLogisticModel:
    def __init__(self, random_state=42, max_iter=1000, class_weight='balanced', C=1, solver= 'lbfgs'):
        # Como parámetros e hiperparámetros por defecto hemos elegido: random_state=42 para que el modelo produzca siempre
        # los mismos resultados ante los mismos datos; max_iter=1000 que establece el número máximo de iteraciones para
        # que el algoritmo converja (por defecto es 100, pero como tenemos bastantes variables después aplicar el
        # OneHotEncoder a lo mejor con 100 no son suficientes y con 1000 le damos más margen al modelo); y
        # class_weight='balanced', que en este caso concreto es muy importante porque al tener tan pocos positivos,
        #  ajusta los pesos dándole más importancia a cada positivo encontrado para compensar su escasez. El
        # hiperparámetro 'C' (regularización) controla cuánto queremos que el modelo aprenda de memoria, cuanto más
        # pequeño es, más regularización y más simple es el modelo. Su valor por defecto es 1. El hiperparámetro
        # 'solver' es el algoritmo de optimización que usa el modelo de forma interna y su valor por defecto es 'lbfgs'.
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter, class_weight=class_weight,
                                        C =C, solver=solver)
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)
    def evaluate(self, X_test, y_test):
        # Para evaluar el modelo, hemos elegido como métricas el recall, F1-score y la matriz de confusión. Accuracy en
        # este caso en que existe un desequilibrio tan grande en las clases podría ser engañosa, ya que si predice
        # siempre un resultado negativo acertará el 97,07 % de las veces (% de resultados negativos del test), pero no
        # detectará los positivos. En cambio, recall mide la proporción de los positivos reales que identificó el modelo,
        # es decir, del total de positivos reales, el modelo ha encontrado un número concreto y esa proporción es el
        # recall. F1_score que es una media entre el recall y la precisión (proporción de los positivos que ha encontrado
        # el modelo, cuáles eran positivos de verdad), pero que penaliza los extremos. 
        prediccion = self.predict(X_test)
        recall = recall_score(y_test, prediccion)
        f1 = f1_score(y_test, prediccion)
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        # La matriz de confusión permite visualizar los cuatro posibles resultados en este caso: verdaderos positivos,
        # verdaderos negativos, falsos positivos y falsos negativos (siendo en este caso lo más relevante los falsos
        # negativos, ya que es el escenario más peligroso desde el punto de vista de la seguridad vial).
        cm = confusion_matrix(y_test, prediccion)
        # Para visualizarla mejor, usamos un mapa de calor de seaborn
        plt.figure(figsize = (8,8))
        sns.heatmap(cm, annot = True, fmt = 'g', linewidths =.5, square = True, cmap ='Blues',
                    xticklabels = ['Negativo', 'Positivo'],
                    yticklabels = ['Negativo', 'Positivo'])
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title('Matriz de confusión')
        plt.show()
    def save_model(self, path='model.pkl'):
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
        print('Modelo guardado correctamente')
    def load_model(self, path='model.pkl'):
        with open(path, 'rb') as file:
            self.model = pickle.load(file)
        print('Modelo cargado correctamente')