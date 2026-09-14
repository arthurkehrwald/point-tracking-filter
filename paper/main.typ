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

= Hintergrund und Motivation <hintergrund>

Diese Arbeit geht aus dem Masterprojekt des Wintersemesters 25/26 hervor, in dem ein interaktives System mit einem autostereoskopischen Display entwickelt wurde.
Zur Anpassung der dargestellten Perspektive an den Betrachtungspunkt haben meine Komilitonen Duy-Anh Do und Randolf Appel eine Trackingsoftware programmiert, die mithilfe einer Stereokamera die Augenpositionen des Betrachters in Echtzeit bestimmt.
Die Software erkennt Augen zuverlässig und liefert Positionen mit geringer Latenz und hoher Frequenz.
Durch das Funktionsprinzip des Displays (Parallaxbarriere) waren die Anforderungen an Genauigkeit, Stabilität, und Latenz der Augenpositionen allerdings zu hoch für unser Trackingsystem.
Bereits Abweichungen um wenige Millimeter führten zu Kanaltrennungsfehlern und mussten durch Verengung der Barriereapertur ausgeglichen werden.
Die kombinierten Latenzen der Stereokamera, der Tracking- und Renderingsoftware und des Displays verzögerten die dargestellte Perspektive außerdem stets um mindestens 100 ms gegenüber den aktuellen Augenpositionen.
Das Endergebnis war ein dunkles, flackerndes Bild mit einer schlechten Tiefenwirkung.
Bei Kopfbewegungen brach die Kanaltrennung völlig zusammen.

= Zielsetzung <ziel>

Ziel ist es, das bestehende Trackingsystem zu vereinfachen, zu überarbeiten und das Endergebnis mithilfe des _Optitrack_ Motion Capture Systems der TH Köln unabhängig von unserem Stereodisplay möglichst objektiv zu bewerten.
Im Vordergrund steht dabei die Glättung und Extrapolation der triangulierten Punkte durch kubische Splineinterpolation.
Dadurch sollen die für Stereo-Trackingsysteme typischen Tiefenfehler geglättet und anstelle der Position zum Zeitpunkt der letzten Aufnahme eine zukünftige Position berechnet werden, sodass ein gegebenenfalls angeschlossenes Display auch während Bewegungen eine annähernd korrekte Perspektive anzeigen kann.

= Überarbeitung der Trackingsoftware <überarbeitung>

Zunächst wird die Software auf das Nötigste reduziert:
Die Erkennung der Augen durch ein Deep Learning Modell und die Verfolgung durch Optical Flow werden durch einen Blob-Detektor ersetzt.
Es handelt sich also nicht mehr um ein markerloses Augentrackingsystem, sondern die Software erkennt stattdessen eine Infrarot-LED.
Das dient nicht nur der Vereinfachung, sondern macht es auch möglich, die selbe LED synchron mit dem _Optitrack_ Motion Capture System wie in @auswertung beschrieben zu erfassen.
Die Stereokamera _Luxonis Oak-D Pro_ kommt weiterhin zum Einsatz, allerdings nach manueller Kalibrierung der intrinsischen und extrinsischen Kameraparameter #ref(<luxonis2026>).
Während des Masterprojekts wurde die werkseitige Kalibrierung verwendet.
Der Code zum Abrufen und Rektifizieren der Bilder sowie zur Triangulation der 3D-Koordinaten wird neu geschrieben, bleibt aber äquivalent.

#figure(
    image("assets/point-tracker-ui.png"),
    caption: "Die Nutzeroberfläche der Trackingsoftware. Links oben Einstellungen, links unten Statusinformationen. Links zur Mitte die beiden Kamerabilder, jeweils mit Fadenkreuz auf der LED. Rechts die 3D-Ansicht. Der rote Punkt visualisiert die berechneten Koordinaten. Unten ein Graph der X- (in rot), Y- (in grün), und Z-Koordinaten (in blau) als Funktion der Zeit."
)<tracker-ui>

Eine wesentliche Neuerung besteht in der Nutzeroberfläche (@tracker-ui). Sie zeigt beide Kamerabilder, visualisiert die gefundenden Koordinaten als Text, in 2D und 3D sowohl als Kurven als auch räumlich und zeigt einige Metriken zu Latenz und Verarbeitungsgeschwindigkeit an.
Diese Hilfsmittel erleichtern es enorm, optimale Parameter für den Blob-Detektor und die Kamera zu finden und gegebenenfalls an die Umgebungsbedingungen und die Leistungsfähigkeit des vewendeten Rechners anzupassen.
Mit der ebenfalls neuen Aufnahmefunktion lassen sich die Koordinaten der LED mitsamt Zeitstempeln zur Weiterverarbeitung und Auswertung im CSV-Format abspeichern.

= Aufnahme <aufnahme>

#figure(
  image("assets/recording-setup.jpg", width: 60%),
  caption: "Die Stereokamera mit Trackingmarkern verbunden mit der Aufnahmesoftware auf meinem Laptop im Motion Capture Studio"
)<recording-setup>

Die Signalverarbeitung soll anhand von authentischen Motion Capture Daten entworfen und getestet werden.
Die Aufnahme findet im Motion Capture Studio am Campus Deutz statt. Dabei wird die LED zusammen mit einer Batterie und einem Schalter an einer Brille befestigt.
Bei den Testdaten handelt es sich also um Kopfbewegungen.
Die LED bewegt sich während der Aufnahmen in dem für die Stereokamera sichtbaren Bereich mit einer Distanz von ca. 30 cm bis 3 m.
Um unterschiedliche Nutzungsszenarien zu simulieren, gibt es jeweils fünf Aufnahmen von 30 Sekunden mit zwei verschiedenen Bewegungsmustern:
Einerseits langsame Bewegungen mit längerem Innehalten, wie sie bei der Nutzung eines Bildschirms auftreten würden und andererseits schnelle, abrupte Bewegungen, die beim Sport oder Tanz vorkommen könnten.

= Zeitliche und räumliche Synchronisation

Alle Bewegungen werden gleichzeitig mit der Stereokamera und dem _Optitrack_-System aufgenommen.
Zur zeitlichen Synchronisierung dient der Zeitpunkt, an dem die LED zum ersten Mal in den beiden Aufnahmem erscheint.
Das funktioniert, weil die LED erst eingeschaltet wird, sobald die Aufnahme beider Systeme bereits läuft und die LED für beide Systeme sichtbar ist.
Ein Nachteil ist, dass mögliche Unterschiede in der Latenz der beiden Aufnahmen dabei ausgeglichen werden, und diese eigentlich interessante Information durch das Fehlen eines einheitlichen Zeitgebers verborgen bleibt.

Drei an der Stereokamera befestigte Trackingmarker ermöglichen eine Erfassung ihrer Position und Ausrichtung durch die _Optitrack_ Kameras relativ zur LED und damit die Abbildug der LED Koordinaten vom Koordinatensystem des einen in das des anderen Kamerasystems durch eine Kombination aus einer Translation und einer Rotation.
Dazu muss zunächst eine Zuordnung der Marker vorgenommen werden. Die Halterung der Trackingmarker ist so konstruiert, dass die Marker ein Dreieck mit drei deutlich verschiedenen Seitenlängen bilden.
Die Rotationsmatrix $bold(R)$ und der Translationsvektor $bold(t)$ bilden die durch das _Optitrack_-System bestimmten Positionen der drei Trackingmarker $bold(p_1), bold(p_2), bold(p_3)$ jeweils auf die Positionen der Trackingmarker im Koordinatensystem der Stereokamera $bold(p'_1), bold(p'_2), bold(p'_3)$ ab, so dass für $n in bold({1, 2, 3})$ gilt $bold(p'_n) = bold(p_n) bold(R) + bold(t)$.

= Auswertung <auswertung>

#bibliography("references.yaml")