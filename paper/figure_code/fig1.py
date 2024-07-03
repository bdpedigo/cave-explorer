# %%
from pathlib import Path

import panel as pn

panel_path = Path("./paper/panels")

pn.extension()

# test panel
panel1 = pn.pane.PNG(panel_path / "inhibition-census-wide.png", width=600)
panel1 = pn.Row(pn.pane.Markdown("# A)"), panel1)
# assemble the figure
fig1 = pn.Column(panel1, width=100)
fig1
