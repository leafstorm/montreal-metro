This is what I'm working on for my PyCon talk.

Right now it just reads a textual system description and draws a map,
because I was fighting kernel panics while I was trying to write this.

To draw a map of the Métro, run:

    python challenge.py montreal.txt | dot -Tpng -omontreal.png
