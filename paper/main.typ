#import "@preview/basic-report:0.5.0": *

#show: it => basic-report(
  doc-category: "Projekt im Modul Angewandte Mathematik",
  doc-title: "Glättung und Extrapolation von Motion Capture Messungen durch kubische Splineinterpolation",
  author: "Arthur Kehrwald (11135125)",
  affiliation: "TH Köln, Master Medientechnologie, Prof. Dr. Stefan Michael Grünvogel",
  logo: image("assets/th-koeln-logo.png", width: 3cm),
  language: "de",
  compact-mode: true,
  it,
)

#let posAcc = [#sym.plus.minus 5 mm]

= Hintergrund <hintergrund>

Diese Arbeit geht aus dem Masterprojekt des Wintersemesters 25/26 hervor, in dem ein interaktives System mit einem autostereoskopischen Display entwickelt wurde.
Zur Anpassung der dargestellten Perspektive an den Betrachtungspunkt haben meine Komilitonen Duy-Anh Do und Randolf Appel eine Trackingsoftware programmiert, die mithilfe einer Stereokamera die Augenpositionen des Betrachters in Echtzeit bestimmt.
Die Software erkennt Augen zuverlässig und liefert Positionen mit geringer Latenz und hoher Frequenz.
Durch das Funktionsprinzip des Displays (Parallaxbarriere) waren die Anforderungen an Genauigkeit, Stabilität, und Latenz der Augenpositionen allerdings zu hoch für unser Trackingsystem.
Bereits Abweichungen um wenige Millimeter führten zu Kanaltrennungsfehlern und mussten durch Verengung der Barriereapertur ausgeglichen werden.
Die kombinierten Latenzen der Stereokamera, der Tracking- und Renderingsoftware und des Displays verzögerten die dargestellte Perspektive außerdem stets um mindestens 100 ms gegenüber den aktuellen Augenpositionen.
Das Endergebnis war ein dunkles, flackerndes Bild mit einer schlechten Tiefenwirkung.
Bei Kopfbewegungen brach die Kanaltrennung völlig zusammen.

= Zielsetzung <ziel>

Ziel ist es, das bestehende Trackingsystem zu vereinfachen, zu überarbeiten und das Endergebnis mithilfe des Optitrack Motion Capture Systems der TH Köln unabhängig von unserem Stereodisplay möglichst objektiv zu bewerten.
Im Vordergrund steht dabei die Glättung und Extrapolation der triangulierten Punkte durch kubische Splineinterpolation.
Dadurch sollen die für Stereo-Trackingsysteme typischen Tiefenfehler geglättet und anstelle der Position zum Zeitpunkt der letzten Aufnahme eine zukünftige Position berechnet werden, sodass ein gegebenfalls angeschlossenes Display auch bei Bewegungen eine annähernd korrekte Perspektive anzeigen kann.

= Überarbeitung der Trackingsoftware <überarbeitung>

#figure(
    image("assets/point-tracker-ui.png"),
    caption: "Die Nutzeroberfläche der Trackingsoftware"
)<tracker-ui>

Zunächst wird die Software auf das Nötigste reduziert:
Die Erkennung der Augen durch ein Deep Learning Modell und die Verfolgung durch Optical Flow weden durch einen Blob-Detektor ersetzt.
Es handelt sich also nicht mehr um ein markerloses Augentrackingsystem, sondern die Software erkennt stattdessen eine Infrarot-LED.
Das dient nicht nur der Vereinfachung, sondern macht es auch möglich, die selbe LED synchron mit dem Optitrack Motion Capture System der TH Köln wie in @auswertung beschrieben zu verfolgen.
Die Stereokamera Luxonis Oak-D Pro kommt weiterhin zum Einsatz.
Der Code zum Abrufen und Rektifizieren der Bilder sowie zur Triangulation der 3D-Koordinaten wird neu geschrieben, bleibt aber äquivalent.
Eine wesentliche Neuerung besteht in einer Nutzeroberfläche (@tracker-ui). Sie zeigt beide Kamerabilder, visualisiert die gefundenden Koordinaten als Text, in 2D und 3D sowohl als Kurven als auch räumlich und zeigt einige Metriken zu Latenz und Verarbeitungsgeschwindigkeit an.
Diese Hilfsmittel erleichtern es enorm, optimale Parameter für den Blob-Detektor und die Kamera zu finden.
Mit der ebenfalls neuen Aufnahmefunktion lassen sich die Koordinaten der LED mitsamt Zeitstempeln zur Weiterverarbeitung und Auswertung als CSV-Datei abspeichern.

= Auswertung <auswertung>
