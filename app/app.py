# AUTOGENERATED! DO NOT EDIT! File to edit: ../app.ipynb.

# %% auto 0
__all__ = ['app', 'hops', 'BinaryMultiply']

# %% ../app.ipynb 1
from flask import Flask
import ghhops_server as hs
import rhino3dm

# %% ../app.ipynb 2
# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)

# %% ../app.ipynb 3
# float multiplication
@hops.component(
    "/binmult",
    inputs=[hs.HopsNumber("A"), hs.HopsNumber("B")],
    outputs=[hs.HopsNumber("Multiply")],
)
def BinaryMultiply(a: float, b: float):
    return a * b

# %% ../app.ipynb 4
# run flask app
if __name__ == '__main__':
    app.run()
