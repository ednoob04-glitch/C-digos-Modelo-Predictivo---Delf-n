import seaborn as sns
import matplotlib.pyplot as plt

print("=== CELDA 2: PROCESAMIENTO DE CADENAS DE MARKOV ===")

# 1. Crear parejas de transición periodo actual -> periodo siguiente
df_validado['ESTADO_ACTUAL'] = df_validado['AUTOMATA_ESTADO']
df_validado['ESTADO_SIGUIENTE'] = df_validado.groupby('ID')['AUTOMATA_ESTADO'].shift(-1)

# Quitamos la última fila de cada alumno (no tiene futuro inmediato)
df_markov = df_validado.dropna(subset=['ESTADO_SIGUIENTE']).copy()

# 2. Matriz de transición probabilística
tabla_frecuencias = pd.crosstab(df_markov['ESTADO_ACTUAL'], df_markov['ESTADO_SIGUIENTE'])
matriz_markov = tabla_frecuencias.div(tabla_frecuencias.sum(axis=1), axis=0)

print(f"• Parejas de transición procesadas: {len(df_markov)}")

# 3. Graficación de la Matriz de Markov
plt.figure(figsize=(13, 8))
sns.heatmap(matriz_markov, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Probabilidades de Transición (Cadenas de Markov) - Fase 3', fontsize=14, pad=15)
plt.ylabel('Estado Actual (t)', fontsize=12)
plt.xlabel('Estado Siguiente (t+1)', fontsize=12)
plt.tight_layout()
plt.show()
