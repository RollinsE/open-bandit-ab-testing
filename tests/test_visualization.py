def test_dark_theme_changes_figure_colours_and_restores_light():
    import pandas as pd
    from src import visualization as V
    ab = pd.DataFrame({"campaign": ["men"], "control_ctr": [0.004], "treatment_ctr": [0.005],
                       "control_ctr_lower": [0.003], "control_ctr_upper": [0.005],
                       "treatment_ctr_lower": [0.004], "treatment_ctr_upper": [0.006],
                       "control_clicks": [40], "control_impressions": [10_000],
                       "treatment_clicks": [50], "treatment_impressions": [10_000]})
    with V.theme("dark"):
        dark = V.create_ab_ctr_plot(ab)
    light = V.create_ab_ctr_plot(ab)
    assert dark.layout.paper_bgcolor == V.THEMES["dark"]["SURFACE"]
    assert light.layout.paper_bgcolor == V.THEMES["light"]["SURFACE"]
