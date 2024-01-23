12.01.2024  Guozheng Zhao
Code Enhancements V2:
1.Image normalization:

   -We improved the image normalization method to accommodate images with different pixel value ranges. Now, images are normalized to the    
   range    from 0 to 1.

2.Comments and readability:

   -We added more comments to make the code easier to understand and use.


20.01.2024 Guozheng Zhao&Shiming Wei

Code-Erklärung
1. **Datenupload und Entpacken**: Der Code lädt ZIP-Dateien hoch und entpackt sie in einen Ordner namens 'images'.
2. **Initialisierung des VGG16-Modells**: VGG16, ein vorab trainiertes Deep-Learning-Modell, wird zur Feature-Extraktion initialisiert.
3. **Bildverarbeitung und Feature-Extraktion**: Bilder werden verarbeitet und deren Merkmale (Features) mit dem VGG16-Modell extrahiert.
4. **Label-Zuweisung**: Jedes Bild wird basierend auf seinem Ordner (z.B. Bahrfeldt, Mehl) einem Label zugeordnet.
5. **Modelltraining und -bewertung**: Ein neuronales Netzwerk wird trainiert, um die Bilder anhand der extrahierten Merkmale zu klassifizieren.

Nutzung des Codes
1. Laden Sie die ZIP-Dateien in Google Colab hoch.
2. Führen Sie den Code aus, um die Bilder zu verarbeiten und das Modell zu trainieren.
3. Überprüfen Sie die Genauigkeit des Modells auf dem Testdatensatz, um die Leistung zu bewerten.

Hinweise
- Der Code muss noch angepasst werden!!!
