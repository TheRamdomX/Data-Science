import os

ruta = r"C:\Users\matia\Desktop\DS\Datos"

for archivo in os.listdir(ruta):
    if archivo.endswith(".csv"):
        path = os.path.join(ruta, archivo)
        
        with open(path, "r", encoding="latin-1") as f:
            contenido = f.read()
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(contenido)

        print(f"[OK] convertido: {archivo}")