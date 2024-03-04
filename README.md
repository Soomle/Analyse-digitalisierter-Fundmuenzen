# Analyse digitalisierter mittelalterlicher Fundmünzen

Das Ziel ist es, eine effektive Methode zur Kategorisierung von Bildern anhand visueller Merkmale zu entwickeln. Wir streben danach, ein Modell zu erstellen, das in der Lage ist, Unterschiede in Farbe, Form, Größe und Textur zu erkennen und diese Merkmale zu nutzen, um Bilder in verschiedene Kategorien einzuteilen.

Datensammlung: 
Dieser Ordner sollte Bildmaterial für die Analyse enthalten, das eine Vielzahl von Bildern umfassen kann, nicht nur Münzen.

Image Detector:
Dieser Ordner sollte das Skript "ImageDetector.py" enthalten, das für die Klassifizierung von Bildern anhand der Merkmale, die durch die im Ordner Merkmalsextraktion definierten Skripte und Funktionen extrahiert wurden, verantwortlich ist. Es ermöglicht die Verwendung mehrerer Modelle, um die Bilder in verschiedene Kategorien einzuteilen, basierend auf den erkannten Unterschieden in Farbe, Form, Größe und Textur.

Vorverarbeitung: 
Dieser Ordner sollte Skripte oder Funktionen enthalten, die die Bilder für die Merkmalsextraktion vorbereiten, indem verschiedene Operationen wie Zuschneiden, Skalieren, Konvertieren, Filtern usw. durchgeführt werden.

Merkmalsextraktion: 
Merkmalsextraktion: Dieser Ordner sollte Skripte oder Funktionen enthalten, die die ausgewählten Merkmale aus den Bildern extrahieren, zu denen Form, Farbe, Textur, Kanten usw. gehören können. Insbesondere beinhaltet er das Skript "ModelGeneration.py", das für die Generierung eines Modells aus einer Reihe von Bildern durch Extraktion und Nutzung dieser Merkmale zuständig ist.

Klassifizierung: 
In diesem Ordner erstellen wir einen geeigneten Algorithmus, der die Bilder anhand ihrer Merkmale in verschiedene Kategorien einteilt, unter Verwendung mehrerer Modelle zur Klassifizierung, wie in der Datei ImageDetector.py angegeben.

Modellgenerierung: 
Anstelle eines Sortiersystems sollte ein System entworfen werden, wie in der Datei ModelGeneration.py beschrieben, um Modelle aus einer Reihe von Bildern zu generieren, die dann zur Klassifizierung neuer Bilder verwendet werden können.



Hinweis:

"ModelGeneration.exe" generiert ein Modell aus einer Reihe von Bildern, indem es den Benutzern ermöglicht, eine ZIP-Datei mit Bildern hochzuladen. Das Programm extrahiert Merkmale aus den Bildern, trainiert ein OneClassSVM-Modell mit diesen Merkmalen und speichert das Modell dann an einem bestimmten Ort. Die Benutzer können über eine grafische Schnittstelle die ZIP-Datei hochladen und den Speicherort für das Modell auswählen.

"ImageDetector.exe" ist eine Anwendung zur Bilderkennung, die es dem Benutzer ermöglicht, mehrere Modelle und ein Bild hochzuladen, um das Bild mithilfe der hochgeladenen Modelle zu klassifizieren. Die Benutzer können über eine grafische Schnittstelle Modelle und das Bild hochladen und dann die Klassifizierungsergebnisse der einzelnen Modelle anzeigen.

