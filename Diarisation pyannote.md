import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HARDLINKS"] = "0"

from pyannote.audio import Pipeline
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import shutil
import json

HUGGINGFACE_TOKEN = "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"
audio_file = "CNEWS.wav"

print("🔄 Chargement du pipeline customisé...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)
print("✅ Pipeline chargé.")

print("🎧 Début de l’analyse du fichier audio...")
diarization = pipeline(audio_file, num_speakers=3)
print("✅ Analyse terminée.")

print("📊 Calcul des temps de parole et détection des overlaps précis...")
segments_by_speaker = defaultdict(list)
interventions_by_speaker = defaultdict(int)
speaking_times = defaultdict(float)
all_segments = []

for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments_by_speaker[speaker].append((turn.start, turn.end))
    interventions_by_speaker[speaker] += 1
    speaking_times[speaker] += turn.end - turn.start
    all_segments.append((turn.start, turn.end, speaker))

def split_overlap_segments(all_segments):
    all_segments.sort()
    output = defaultdict(list)
    for i, (start_i, end_i, speaker_i) in enumerate(all_segments):
        overlap_parts = []
        for j, (start_j, end_j, speaker_j) in enumerate(all_segments):
            if speaker_i == speaker_j:
                continue
            if start_j < end_i and end_j > start_i:
                overlap_start = max(start_i, start_j)
                overlap_end = min(end_i, end_j)
                overlap_parts.append((overlap_start, overlap_end))
        if not overlap_parts:
            output[speaker_i].append((start_i, end_i, False))
        else:
            points = [start_i] + [s for s, e in overlap_parts] + [e for s, e in overlap_parts] + [end_i]
            points = sorted(set([p for p in points if start_i <= p <= end_i]))
            for k in range(len(points) - 1):
                seg_start, seg_end = points[k], points[k + 1]
                is_overlap = any(s < seg_end and e > seg_start for s, e in overlap_parts)
                output[speaker_i].append((seg_start, seg_end, is_overlap))
    return output

split_segments = split_overlap_segments(all_segments)

cmap = plt.get_cmap("tab20")
sorted_speakers = sorted(speaking_times.items(), key=lambda x: x[1], reverse=True)
speaker_positions = {speaker: i for i, (speaker, _) in enumerate(sorted_speakers)}
speaker_colors_plotly = {
    speaker: f"rgba({int(255*r)},{int(255*g)},{int(255*b)},0.8)"
    for i, (speaker, _) in enumerate(sorted_speakers)
    for r, g, b, _ in [cmap(i)]
}
speaker_colors_matplotlib = {
    speaker: cmap(i)[:3]
    for i, (speaker, _) in enumerate(sorted_speakers)
}

print("\n📢 Temps de parole par locuteur :")
for speaker, duration in speaking_times.items():
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"{speaker}: {minutes} min {seconds} sec")

print("📊 Création du graphe interactif (zoomable)...")
plotly_fig = go.Figure()
bar_height = 0.4

for speaker, segs in split_segments.items():
    y_base = speaker_positions[speaker]
    for start, end, is_overlap in segs:
        y0 = y_base - bar_height
        y1 = y_base + bar_height
        color = 'rgba(255,0,0,0.6)' if is_overlap else speaker_colors_plotly[speaker]
        duration = int(end - start)
        hover = f"{speaker} - {int(start//60)}:{int(start%60):02} - {int(end//60)}:{int(end%60):02} - {duration} sec"
        plotly_fig.add_trace(go.Scatter(
            x=[start, end, end, start, start],
            y=[y0, y0, y1, y1, y0],
            fill="toself",
            mode="lines",
            line=dict(width=0),
            fillcolor=color,
            hoveron='fills',
            name=speaker,
            hovertext=hover,
            showlegend=False
        ))

# Légende Overlap
plotly_fig.add_trace(go.Scatter(
    x=[0, 1, 1, 0, 0],
    y=[len(speaker_positions)+1]*5,
    fill='toself',
    mode='lines',
    fillcolor='rgba(255,0,0,0.6)',
    line=dict(width=0),
    name="Overlap détecté",
    showlegend=True
))

# Format X en min:sec
max_time = max(end for start, end, _ in all_segments)
x_tickvals = list(range(0, int(max_time) + 60, 60))
x_ticktext = [f"{v//60}:{v%60:02}" for v in x_tickvals]

y_tickvals = list(speaker_positions.values()) + [len(speaker_positions)+1]
y_ticktext = list(speaker_positions.keys()) + ["Overlap"]

# Légende horizontale unique temps de parole
legend_text = " | ".join(
    f"{speaker} : {int(speaking_times[speaker] // 60)} min {int(speaking_times[speaker] % 60)} sec"
    for speaker, _ in sorted_speakers
)

plotly_fig.update_layout(
    title="Timeline des prises de parole (interactif)",
    xaxis=dict(title="Temps (min:sec)", tickmode='array', tickvals=x_tickvals, ticktext=x_ticktext),
    yaxis=dict(title="Locuteurs", tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext, automargin=True),
    height=400 + 40 * len(speaker_positions),
    margin=dict(l=60, r=30, t=60, b=150),  # ✅ marge basse plus grande
    showlegend=True
)

plotly_fig.add_annotation(
    text=f"<b>Temps de parole total</b><br>{legend_text}",
    xref="paper", yref="paper",
    x=0.5, y=-0.3, showarrow=False,
    font=dict(size=12),
    align="center"
)

plotly_fig.write_html("timeline_parole_interactif.html")
print("✅ Graphe interactif enregistré sous 'timeline_parole_interactif.html'")

with open("timeline_parole_interactif.json", "w") as f:
    json.dump(plotly_fig.to_plotly_json(), f)
print("✅ Fichier JSON généré sous 'timeline_parole_interactif.json'")

print("📊 Création de l’histogramme des interventions...")
plt.figure(figsize=(8, 4))
speakers = list(interventions_by_speaker.keys())
counts = list(interventions_by_speaker.values())
colors = [speaker_colors_matplotlib[speaker] for speaker in speakers]

plt.bar(speakers, counts, color=colors)
plt.xlabel("Locuteurs")
plt.ylabel("Nombre d’interventions")
plt.title("Histogramme des interventions par locuteur")
for i, speaker in enumerate(speakers):
    plt.text(i, counts[i] + 0.5, f"{counts[i]}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("interventions_par_locuteur.png")
print("✅ Histogramme enregistré sous 'interventions_par_locuteur.png'")
plt.show()

react_public_dir = os.path.join(os.path.expanduser("~"), "Documents", "timeline-diarisation", "public")
shutil.copy("timeline_parole_interactif.html", os.path.join(react_public_dir, "timeline_parole_interactif.html"))
shutil.copy("timeline_parole_interactif.json", os.path.join(react_public_dir, "timeline_parole_interactif.json"))
print("✅ Fichier HTML et JSON copiés automatiquement dans le dossier React.")
