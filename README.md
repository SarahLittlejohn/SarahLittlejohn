<img src="qubit-snakes.svg" width="100%" alt="A cats and ladders board drawn as IBM's Fez quantum chip, showing today's quantum dice roll and where the pawn landed.">

A game that plays itself, one move a day, on the qubits of IBM's Fez processor. Every circle is a qubit, numbered in order of play. The ladders are the bridge qubits that link one row of the chip to the next: you can climb one if both of its couplers have a CZ error below 1% and there's no cat on the bridge qubit. Noisy qubits are Schrödinger's cats: land on a cat (readout error above 2%) and you go back 3, or on a kitten (1.2% to 2%) and you go back 2.

Every morning at 07:30 UTC a [GitHub Action](.github/workflows/qubit-snakes.yml) runs [qubit_snakes.py](qubit_snakes.py), which rolls the dice by putting three qubits into superposition on a real IBM Quantum processor and measuring them.
