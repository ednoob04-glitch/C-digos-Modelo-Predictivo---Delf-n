from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("=== CELDA 3: INTELIGENCIA ARTIFICIAL BALANCEADA (CORREGIDA) ===")

# =====================================================================
# 1. DEFINICIÓN DEL DESTINO FINAL SEGÚN ESTADOS UNIVERSITARIOS REALES
# =====================================================================
ultimo_estado_por_alumno = df_validado.sort_values('PERIODO').groupby('ID')['AUTOMATA_ESTADO'].last()

# Definimos la regla de Éxito basada en tus datos reales de la Celda 1
estados_exito_real = ['Continuo regular', 'Primera vez en una carrera', 'Aspirante inscrito']
desenlace_exito = ultimo_estado_por_alumno.apply(lambda x: 1 if x in estados_exito_real else 0)

# Mapeamos este destino definitivo (0 o 1) a todos los semestres históricos del alumno
df_validado['DESENLACE_FINAL'] = df_validado['ID'].map(desenlace_exito)

# Duplicamos la base para trabajar el modelo a largo plazo
df_largo_plazo = df_validado.copy()

print(f"• Alumnos totales analizados para el futuro: {df_largo_plazo['ID'].nunique()}")
print(f"• Alumnos identificados en trayectoria de Éxito: {(desenlace_exito == 1).sum()}")
print(f"• Alumnos identificados en trayectoria de Riesgo: {(desenlace_exito == 0).sum()}")


# =====================================================================
# 2. DIVISIÓN ESTRATIFICADA 80/20 POR ALUMNO
# =====================================================================
df_alumnos_unicos = df_largo_plazo.drop_duplicates(subset=['ID'])
ids_unicos = df_alumnos_unicos['ID'].values
destinos_unicos = df_alumnos_unicos['DESENLACE_FINAL'].values

ids_train, ids_test = train_test_split(
    ids_unicos, test_size=0.20, random_state=42, stratify=destinos_unicos
)

df_train_lp = df_largo_plazo[df_largo_plazo['ID'].isin(ids_train)].reset_index(drop=True)
df_test_lp = df_largo_plazo[df_largo_plazo['ID'].isin(ids_test)].reset_index(drop=True)


# =====================================================================
# 3. PREPARACIÓN DE MATRICES
# =====================================================================
variables_entrada = ['AUTOMATA_ESTADO', 'PPP', 'PPA', 'NRO_CURSOS_APROBADOS', 'CREDITOS_APROBADOS']

X_train_lp = pd.get_dummies(df_train_lp[variables_entrada], columns=['AUTOMATA_ESTADO'])
y_train_lp = df_train_lp['DESENLACE_FINAL']
X_test_lp = pd.get_dummies(df_test_lp[variables_entrada], columns=['AUTOMATA_ESTADO'])
y_test_lp = df_test_lp['DESENLACE_FINAL']

X_train_lp, X_test_lp = X_train_lp.align(X_test_lp, join='left', axis=1, fill_value=0)


# =====================================================================
# 4. ENTRENAMIENTO DEL CLASIFICADOR (Con pesos balanceados)
# =====================================================================
print("• Entrenando el modelo RandomForest para predicción de éxito futuro...")
modelo_graduacion = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo_graduacion.fit(X_train_lp, y_train_lp)


# =====================================================================
# 5. PREDICCIÓN CON UMBRAL CALIBRADO (Aquí ocurre el balanceo)
# =====================================================================
# Obtenemos las probabilidades en lugar de la predicción directa
probabilidades = modelo_graduacion.predict_proba(X_test_lp)

# Definimos el umbral del 30% (0.30)
# Si la probabilidad de ÉXITO (columna 1) es menor a 0.70, clasificamos como RIESGO (0)
nuevo_umbral_exito = 0.70 
y_pred_lp = np.where(probabilidades[:, 1] >= nuevo_umbral_exito, 1, 0)


# =====================================================================
# 6. EVALUACIÓN Y GRÁFICA NARANJA BALANCEADA
# =====================================================================
print(f"\nSISTEMA PREDICTIVO COMPLETADO (UMBRAL OPTIMIZADO AL 30% DE RIESGO)")
print(f"Precisión del Pronóstico a Largo Plazo (Accuracy): {accuracy_score(y_test_lp, y_pred_lp):.2%}")

print("\n=== REPORTES METODOLÓGICOS PARA LA TESIS ===")
print(classification_report(y_test_lp, y_pred_lp, zero_division=0, target_names=['Trayectoria en Riesgo', 'Trayectoria de Éxito']))

# Matriz de confusión naranja balanceada
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_lp, y_pred_lp), annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Predicho: Riesgo', 'Predicho: Éxito'], 
            yticklabels=['Real: Riesgo', 'Real: Éxito'])
plt.title('Matriz de Confusión Calibrada: Predicción de Éxito a Largo Plazo', fontsize=12, pad=15)
plt.ylabel('Destino Real Final', fontsize=10)
plt.xlabel('Destino Predicho por la IA', fontsize=10)
plt.tight_layout()
plt.show()
