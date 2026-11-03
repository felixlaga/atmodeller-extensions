# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained reader for the computed cooling-path CSV."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    paths = pd.read_csv(HERE / "outputs/cooling_paths.csv")
    onsets = pd.read_csv(HERE / "outputs/onsets.csv")
    data = {}
    for p in (2, 5, 10):
        for w in (1, 3, 5):
            d = paths.loc[
                (paths.model == "gilmore26")
                & (paths.pressure_GPa == p)
                & (paths.hydrogen_wt_percent == w)
            ]
            b = onsets.loc[
                (onsets.model == "gilmore26")
                & (onsets.pressure_GPa == p)
                & np.isclose(onsets.hydrogen_wt_percent, w)
            ].iloc[0]
            ts = b.spinodal_temperature_K
            data[f"{p}_{w}"] = {
                "tb": float(b.binodal_temperature_K),
                "ts": float(ts) if np.isfinite(ts) else None,
                "rows": d[["temperature_K", "H_equilibrium_fraction", "H_delayed_limit_fraction"]]
                .round(8)
                .values.tolist(),
            }
    html = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hydrogen separation during cooling</title><style>
body{margin:0;color:var(--foreground,#203b49);font-family:var(--font-sans,system-ui);font-size:14px;line-height:1.5}*{box-sizing:border-box}
p{margin:0 0 14px} .lead{font-size:19px;font-weight:650;line-height:1.3;margin-bottom:8px}.muted{color:var(--muted-foreground,#546974)}
.controls{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}label{display:grid;gap:5px;font-size:12px;font-weight:600}
select{font:inherit;font-size:14px;min-width:135px;padding:8px;border:1px solid var(--border,#ccd8dc);border-radius:8px;background:var(--background,white);color:inherit}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12px}.key{display:inline-block;width:24px;margin-right:6px;vertical-align:middle;border-top:3px solid var(--chart-1,#166b79)}.key.delayed{border-color:var(--chart-2,#a45a32);border-top-style:dashed}
svg{display:block;width:100%;height:300px;overflow:visible}svg text{fill:currentColor;font-size:11px;font-family:inherit}
.readout{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.stat{background:var(--secondary,#edf4f4);border-radius:var(--radius,10px);padding:16px}.value{font-size:26px;font-weight:600;line-height:1.3}.small{font-size:12px}
input{width:100%;accent-color:var(--chart-1,#166b79)}.note{font-size:12px;margin-top:12px;padding-left:12px;border-left:3px solid var(--border,#cbd7dc)}
@media(max-width:400px){.stat{padding:16px}.value{font-size:23px}.controls{gap:12px}select{min-width:125px}}
</style></head><body>
<p class="lead">How much hydrogen separation could be delayed?</p>
<p class="muted">A closed H₂–MgSiO₃ parcel, using the final 2026 fit with its composition basis preserved. Explore two limiting cooling histories.</p>
<div class="controls"><label>Pressure<select id="pressure"><option value="2">2 GPa</option><option value="5">5 GPa</option><option value="10">10 GPa</option></select></label>
<label>Local hydrogen by mass<select id="weight"><option value="1">1 wt%</option><option value="3" selected>3 wt%</option><option value="5">5 wt%</option></select></label></div>
<div class="legend"><span><i class="key"></i>Instant equilibrium</span><span><i class="key delayed"></i>No nucleation until spinodal</span></div>
<svg id="plot" role="img" aria-label="Hydrogen fraction in the separate H-rich phase as the parcel cools"></svg>
<label>Inspect temperature: <span id="temperatureLabel">3000 K</span><input id="temperature" type="range" min="2000" max="4000" step="5" value="3000" aria-label="Temperature in kelvin"></label>
<div class="readout"><div class="stat"><div class="small muted">Equilibrium H in rich phase</div><div class="value" id="eq">—</div></div><div class="stat"><div class="small muted">Delayed-limit H in rich phase</div><div class="value" id="delay">—</div></div></div>
<p class="small" id="boundaries"></p>
<p class="note muted"><b>Inventory bound, not an atmospheric prediction.</b> All hydrogen remains in the parcel. The dashed history imposes no nucleation until instability, followed by immediate equilibration; no physical rate is calculated. Below 2500 K is an extrapolation below the lowest tabulated two-phase simulation temperature. Published coexistence data do not yet tightly validate this fit.</p>
<script>const data=__DATA__;const $=id=>document.getElementById(id);let checks=0;
function draw(){const p=$('pressure').value,w=$('weight').value,s=data[p+'_'+w],t=+$('temperature').value,svg=$('plot');const W=Math.max(320,svg.getBoundingClientRect().width),H=300,L=42,R=12,Top=28,Bottom=45;
const x=v=>L+(4000-v)/2000*(W-L-R),y=v=>H-Bottom-v*(H-Top-Bottom);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
const path=col=>s.rows.map((r,i)=>(i?'L':'M')+x(r[0]).toFixed(2)+','+y(r[col]).toFixed(2)).join(' ');
let a=`<text x="${L}" y="15">Hydrogen in separate H-rich phase (%)</text><rect x="${x(2500)}" y="${Top}" width="${x(2000)-x(2500)}" height="${H-Top-Bottom}" fill="var(--warning,#ce9c43)" opacity=".10"/>`;
for(const v of [0,.25,.5,.75,1])a+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="var(--border,#dce3e6)"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end">${100*v}</text>`;
for(const v of [4000,3500,3000,2500,2000])a+=`<text x="${x(v)}" y="${H-Bottom+21}" text-anchor="${v===2000?'end':v===4000?'start':'middle'}">${v}</text>`;
a+=`<text x="${(W+L-R)/2}" y="${H-3}" text-anchor="middle">Cooling temperature (K) →</text><path d="${path(1)}" fill="none" stroke="var(--chart-1,#166b79)" stroke-width="2.5"/><path d="${path(2)}" fill="none" stroke="var(--chart-2,#a45a32)" stroke-width="2.5" stroke-dasharray="7 5"/><line x1="${x(t)}" x2="${x(t)}" y1="${Top}" y2="${H-Bottom}" stroke="var(--muted-foreground,#687a84)" stroke-dasharray="3 3"/>`;
const r=s.rows[Math.round((4000-t)/5)];for(const [col,color] of [[1,'--chart-1'],[2,'--chart-2']])a+=`<circle cx="${x(t)}" cy="${y(r[col])}" r="4" fill="var(${color},#166b79)"/>`;svg.innerHTML=a;
$('temperatureLabel').textContent=t+' K';$('eq').textContent=(100*r[1]).toFixed(1)+'%';$('delay').textContent=(100*r[2]).toFixed(1)+'%';$('boundaries').textContent=`Equilibrium separation: ${s.tb.toFixed(0)} K. `+(s.ts===null?'No positive spinodal temperature in this fit for this composition.':`Spinodal: ${s.ts.toFixed(0)} K. Available metastable interval: ${(s.tb-s.ts).toFixed(0)} K.`);
if(!Number.isFinite(r[1])||r[1]<0||r[1]>1||r[2]>r[1]+1e-8)throw Error('Invalid plotted inventory');checks++;
}
for(const p of ['2','5','10'])for(const w of ['1','3','5']){$('pressure').value=p;$('weight').value=w;draw();} $('pressure').value='2';$('weight').value='3';draw();console.log('PASS: '+checks+' rendered-case checks; 9 computed cooling paths.');
for(const id of ['pressure','weight','temperature'])$(id).addEventListener('input',draw);new ResizeObserver(draw).observe($('plot'));
</script></body></html>""".replace(
        "__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False)
    )
    (HERE / "explorer.html").write_text(html)
    print("Created explorer.html from 9 computed cooling paths")


if __name__ == "__main__":
    main()
