import pandas as pd
import numpy as np

print("=== CELDA 1: LIMPIEZA INSTITUCIONAL (VERSIÓN DEFINITIVA) ===")

# 1. Carga del archivo original
df_raw = pd.read_excel('DATOS_NUEVO.xlsx')
print(f"• Registros iniciales en bruto: {len(df_raw)}")

# 2. Selección de columnas usando los nombres exactos de tu captura
columnas_necesarias = [
    'ID', 'PERIODO', 'PROGRAMA', 'AUTOMATA_ESTADO',
    'PROMEDIO', 'PROMEDIO_ACUMULADO', 'NRO_CURSOS_APROBADOS', 'CREDITOS_APROBADOS'
]
df_subset = df_raw[columnas_necesarias].copy()

# Renombramos promedios para el estándar de tu tesis
df_subset = df_subset.rename(columns={
    'PROMEDIO': 'PPP', 
    'PROMEDIO_ACUMULADO': 'PPA'
})

# Limpieza profunda de espacios ocultos en la columna de estados
df_subset['AUTOMATA_ESTADO'] = df_subset['AUTOMATA_ESTADO'].astype(str).str.strip()

# 3. Eliminación de registros vacíos en variables críticas
columnas_criticas = ['AUTOMATA_ESTADO', 'PPP', 'PPA', 'CREDITOS_APROBADOS', 'NRO_CURSOS_APROBADOS']
df_sin_vacios = df_subset.dropna(subset=columnas_criticas).copy()
df_ordenado = df_sin_vacios.sort_values(by=['ID', 'PERIODO']).reset_index(drop=True)

# 4. Diccionario de transiciones legales del autómata
transiciones_legales = {
    'Transferencia externa': ['Aspirante inscrito'],
    'Transferencia interna': ['Primera vez en una carrera'],
    'Aspirante inscrito': ['Primera vez en una carrera'],
    'Primera vez en una carrera': ['Continuo regular', 'PAP', 'Transferencia interna', 'PFU'],
    'Continuo regular': ['Continuo regular', 'PAP', 'Grado', 'Transferencia interna', 'PFU', 'Reinicio'],
    'PAP': ['Continuo regular', 'PAT', 'Transferencia interna'],
    'PAT': ['Continuo regular', 'Recuperación académica'],
    'Recuperación académica': ['Continuo regular', 'Exclusión', 'Reinicio'],
    'Exclusión': ['Reinicio'],
    'Reinicio': ['Primera vez en una carrera', 'Continuo regular'],
    'PFU': ['PFU', 'Reingreso', 'Reinicio'],
    'Reingreso': ['Continuo regular'],
    'Grado': [] 
}

ids_trayectorias_corruptas = set()

# Auditoría de secuencias por alumno
for estudiante_id, historial in df_ordenado.groupby('ID'):
    secuencia_estados = historial['AUTOMATA_ESTADO'].tolist()
    for i in range(len(secuencia_estados) - 1):
        actual = secuencia_estados[i]
        siguiente = secuencia_estados[i+1]
        
        # Regla de escape: Si el estado actual contiene 'Grado', la secuencia es exitosa y se salva el alumno
        if 'grado' in actual.lower():
            break
        if actual not in transiciones_legales or siguiente not in transiciones_legales[actual]:
            ids_trayectorias_corruptas.add(estudiante_id)
            break

# Base de datos purificada oficial final
df_validado = df_ordenado[~df_ordenado['ID'].isin(ids_trayectorias_corruptas)].copy()

print("\n=== REPORTE FINAL DE AUDITORÍA ===")
print(f"✔ Estudiantes descartados por saltos inválidos: {len(ids_trayectorias_corruptas)}")
print(f"✔ Estudiantes 100% LÓGICOS retenidos: {df_validado['ID'].nunique()}")
print(f"✔ Filas con la palabra 'Grado' que logramos salvar: {df_validado['AUTOMATA_ESTADO'].str.contains('Grado', case=False).sum()}")
