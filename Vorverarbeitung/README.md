09.01.2024  Shiming Wei

Dieser Teil des Codes ist hauptsächlich für das Sammeln und Vorverarbeiten von Daten zuständig.

1. Importieren von notwendigen Bibliotheken:
   - `cv2`: Eine Bibliothek für Bildverarbeitung (OpenCV).
   - `numpy`: Eine Bibliothek für mathematische Operationen und Datenmanipulation.
   - `zipfile`: Eine Bibliothek zur Handhabung von ZIP-Dateien.
   - `os`: Bietet Funktionen zur Interaktion mit dem Betriebssystem.
   - `VGG16, preprocess_input`: Importiert das VGG16-Modell und die zugehörige Vorverarbeitungsfunktion aus dem TensorFlow Keras-Modul.
   - `files`: Ein Modul von Google Colab für den Datei-Upload.
   - `matplotlib.pyplot`: Eine Bibliothek für Datenvisualisierung.

2. Upload und Entpacken der Daten:
   - Mit der Funktion `files.upload()` werden ZIP-Dateien in Google Colab hochgeladen.
   - Durchlaufen der hochgeladenen Dateien und Entpacken in den Ordner 'images' mit `zipfile.ZipFile`.

3. Initialisierung des VGG16-Modells:
   - Erstellen eines VGG16-Modells mit der Funktion `VGG16`, wobei `weights='imagenet'` und `include_top=False` gesetzt werden, um das Modell mit den ImageNet-Gewichten ohne die obersten Verbindungsschichten zu verwenden.

4. Definition der Bildverarbeitungsfunktion `process_image`:
   - Lädt eine Bilddatei.
   - Überprüft, ob das Bild gültig ist, und gibt bei Ungültigkeit None zurück.
   - Konvertiert das Bild in das RGB-Format, falls es drei Kanäle hat.
   - Skaliert das Bild auf die Größe von 224x224 (die Eingabegröße, die das VGG16-Modell erfordert).
   - Normalisiert das Bild und führt eine Vorverarbeitung mit `preprocess_input` durch.

5. Definition der Funktion `explore_features` zur Erkundung von Merkmalen:
   - Berechnet und druckt den Durchschnitt und die Standardabweichung der Merkmale.

6. Definition der Funktion `visualize_image` zur Bildvisualisierung:
   - Normalisiert oder kürzt das Bild je nach Datentyp, um sicherzustellen, dass die Pixelwerte im korrekten Bereich liegen.
   - Stellt das Bild mit `matplotlib` dar.

7. Verarbeitung und Visualisierung von Bildern:
   - Durchläuft alle Bilder im Ordner 'images'.
   - Lädt und zeigt das Originalbild.
   - Verarbeitet das Bild mit `process_image`.
   - Falls das Bild gültig ist, extrahiert es Merkmale mit dem VGG16-Modell und erkundet und zeigt das verarbeitete Bild mit `explore_features` und `visualize_image`.
  
Die möglichen Probleme im Code sind falsche Vorverarbeitung der Bilder und nicht abgestimmte Modellgewichte, die zu Nullen in den Feature-Matrizen führen, sowie eine potenzielle fehlerhafte Normalisierung oder Skalierung, die das Auftreten eines schwarzen Bildes verursachen kann.

12.01.2024  Guozheng Zhao
Code Enhancements V2:
1.Image normalization:

   -We improved the image normalization method to accommodate images with different pixel value ranges. Now, images are normalized to the    
   range    from 0 to 1.

2.Comments and readability:

   -We added more comments to make the code easier to understand and use.
