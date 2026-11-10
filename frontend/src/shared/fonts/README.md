# Bundled fonts

Figtree is the UI and heading face, Aleo is the speech/transcript face, and Cousine
is the code and keyboard-hint face. All files are served locally through Vite.
Each family includes its upstream SIL Open Font License in `LICENSE-<family>.txt`.
Cousine's current upstream release is OFL, replacing its older Apache release.

Sources: [google/fonts](https://github.com/google/fonts/tree/6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec/ofl),
revision `6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec`:

- `figtree/Figtree[wght].ttf` and `figtree/Figtree-Italic[wght].ttf` (weights 300 to 900).
- `aleo/Aleo[wght].ttf` and `aleo/Aleo-Italic[wght].ttf` (weights 100 to 900).
- `cousine/Cousine-{Regular,Bold,Italic,BoldItalic}.ttf` (weights 400 and 700).

The complete upstream fonts were converted to WOFF2 with fontTools 4.68.0 and
Brotli 1.2.1, preserving every glyph, layout table, and variable weight axis.
No character subsetting was applied: names and transcript text can use the full
upstream coverage, and unsupported characters use the theme's fallback stacks.
Browsers fetch only the styles used on a page. The eight files total about 634 KiB.

To reproduce a file from its upstream TTF with those fontTools/Brotli versions:

```python
from fontTools.ttLib import TTFont

font = TTFont("Figtree[wght].ttf")
font.flavor = "woff2"
font.save("Figtree-Variable.woff2")
```
