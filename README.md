# circuit_recognizer

Circuit simulation programs are very efficient and fast way of observing behaviors of electrical circuits. They are fundamental tools for engineers to analyze characteristics of circuits and get simulation results. To use the benefits of simulation programs, engineers need circuit schematic which contains components and interconnections between them. The simulation programs provide a graphical interface to create circuit schematics. However, it’s popular to create initial concepts of electrical circuits using pencil and paper for engineers. Because of that, engineers spend additional time to transform hand-drawn circuit to required schematic file of simulation program. 

This project explains a method to create schematic file from hand-drawn circuit sketch. It segments components and connection wires using combination of end point analysis and line detection, then it describes component region using HOG features. It recognizes components using SVM classification. 


