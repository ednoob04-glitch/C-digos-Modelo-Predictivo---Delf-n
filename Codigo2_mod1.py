import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("=== CELDA 2 REPOTENCIADA: PROCESAMIENTO DE CADENAS DE MARKOV SELECCIONADO ===")

# =====================================================================
# 1. PREPARACIÓN DE DATOS (Parejas de Transición)
# =====================================================================
# Usamos tu columna 'AUTOMATA_ESTADO' para crear el estado siguiente
df_validado['ESTADO_ACTUAL'] = df_validado['AUTOMATA_ESTADO']
df_validado['ESTADO_SIGUIENTE'] = df_validado.groupby('ID')['AUTOMATA_ESTADO'].shift(-1)

# Calculamos dinámicamente la cohorte (mínimo PERIODO de cada alumno)
periodo_ingreso = df_validado.groupby('ID')['PERIODO'].min()
df_validado['COHORTE_CALCULADA'] = df_validado['ID'].map(periodo_ingreso)

# Quitamos la última fila de cada alumno (la que no tiene futuro inmediato en los datos)
df_markov = df_validado.dropna(subset=['ESTADO_SIGUIENTE']).copy()

print(f"• Parejas de transición totales procesadas: {len(df_markov)}")


# =====================================================================
# FUNCIÓN AUXILIAR PARA CALCULAR Y GRAFICAR MARKOV
# =====================================================================
def generar_matriz_markov(df_origen, titulo_grafica, color_map='Blues'):
    """
    Toma un subconjunto de datos, calcula la matriz de frecuencias,
    la normaliza a probabilidades de transición y la grafica en un mapa de calor.
    """
    if df_origen.empty:
        print(f"  [!] No hay suficientes datos para generar: {titulo_grafica}")
        return
        
    # Calcular matriz de frecuencias y normalizar por filas para obtener probabilidades
    tabla_frecuencias = pd.crosstab(df_origen['ESTADO_ACTUAL'], df_origen['ESTADO_SIGUIENTE'])
    matriz_prob = tabla_frecuencias.div(tabla_frecuencias.sum(axis=1), axis=0)
    
    # Graficar
    plt.figure(figsize=(11, 7))
    sns.heatmap(matriz_prob, annot=True, cmap=color_map, fmt='.2f', linewidths=0.5, cbar=True)
    plt.title(titulo_grafica, fontsize=12, pad=15, fontweight='bold')
    plt.ylabel('Estado Actual (t)', fontsize=10)
    plt.xlabel('Estado Siguiente (t+1)', fontsize=10)
    plt.tight_layout()
    plt.show()


# =====================================================================
# A. MATRIZ GLOBAL (Toda la Universidad)
# =====================================================================
print("\n• Generando Matriz Global...")
generar_matriz_markov(
    df_markov, 
    'Matriz Global de Probabilidades de Transición (Cadenas de Markov) - General',
    color_map='Blues'
)


# =====================================================================
# B. MATRICES POR PROGRAMA (Únicamente 3 seleccionados)
# =====================================================================
print("\n• Generando Matrices para Programas Seleccionados...")

# MODIFICA ESTA LISTA: Escribe aquí los nombres exactos de tus 3 carreras
carreras_seleccionadas = [
    'INGENIERIA DE SISTEMAS', 
    'ADMINISTRACION DE EMPRESAS', 
    'PSICOLOGIA'
]

for carrera in carreras_seleccionadas:
    # Verificamos si la carrera existe en la base de datos
    if carrera in df_markov['PROGRAMA'].values:
        print(f"  -> Procesando programa seleccionado: {carrera}")
        df_filtrado_programa = df_markov[df_markov['PROGRAMA'] == carrera]
        
        generar_matriz_markov(
            df_filtrado_programa, 
            f'Matriz de Transición de Markov - Programa: {carrera}',
            color_map='YlGnBu' 
        )
    else:
        print(f"  [!] Alerta: La carrera '{carrera}' no se encontró en la columna 'PROGRAMA'.")
        print("      Carreras disponibles en tu base para copiar exactamente:")
        print(df_markov['PROGRAMA'].dropna().unique())


# =====================================================================
# C. MATRICES POR COHORTE (Usando la cohorte calculada dinámicamente)
# =====================================================================
print("\n• Generando Matrices por Cohorte de Ingreso...")
cohortes = sorted(df_markov['COHORTE_CALCULADA'].dropna().unique())

# Graficamos las últimas 3 cohortes para mantener todo bien sintetizado
ultimas_cohortes = cohortes[-3:] 

for cohorte in ultimas_cohortes:
    print(f"  -> Procesando cohorte: {cohorte}")
    df_filtrado_cohorte = df_markov[df_markov['COHORTE_CALCULADA'] == cohorte]
    
    generar_matriz_markov(
        df_filtrado_cohorte, 
        f'Matriz de Transición de Markov - Cohorte de Ingreso: {cohorte}',
        color_map='Purples' 
    )
