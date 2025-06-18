# pip install pyanatomogram
import pyanatomogram as pa

# 1. Load the human anatomogram
anat = pa.Anatomogram('homo_sapiens')

# 2. Optionally, list available tissue names to find correct keys:
# print(anat.get_tissue_names())

# 3. Define styles for each part
tissue_styles = {
    'brain':           {'fill': 'red'},      # head & neck proxy
    'urinary_bladder': {'fill': 'purple'},
    'vagina':          {'fill': 'pink'},
    'uterus':          {'fill': 'magenta'},
    'testis':          {'fill': 'blue'},
    'skeletal_muscle': {'fill': 'green'},    # highlights muscle (arms & legs)
}

# 4. Apply styles (missing tissues will be skipped)
for tissue, style in tissue_styles.items():
    if tissue in anat.get_tissue_names():
        anat.set_tissue_style(tissue, **style)

# 5. Render via Matplotlib
fig = anat.to_matplotlib()
fig.show()
