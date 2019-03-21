import matchzoo as mz

from matching.load_data import load_data
from matching.train_test_split import train_test_split


def plot_results(results_show):
    import bokeh
    from bokeh.plotting import figure
    from bokeh.io import output_file, show
    from bokeh.layouts import column
    from bokeh.models.tools import HoverTool

    charts = {
        metric: figure(
            title=str(metric),
            sizing_mode='scale_width',
            width=800, height=400
        ) for metric in results_show[0]['history'].history.keys()
    }
    hover_tool = HoverTool(tooltips=[
        ("x", "$x"),
        ("y", "$y")
    ])
    for metric, sub_chart in charts.items():
        lines = {}
        for result, color in zip(results_show, bokeh.palettes.Category10[10]):
            x = result['history'].epoch
            y = result['history'].history[metric]
            lines[result['name']] = sub_chart.line(
                x, y, color=color, line_width=2, alpha=0.5,
                legend=result['name'])
            sub_chart.add_tools(hover_tool)

    output_file('results.html')
    show(column(*charts.values()))


def auto_results(train_data_pack, test_data_pack):
    model_classes = [
        mz.models.CDSSM,
        mz.models.DSSM,
        mz.models.DUET,
    ]
    task = mz.tasks.Ranking(metrics=['ap', 'ndcg'])
    results = []
    for model_class in model_classes:
        print(model_class)
        model = model_class()
        model.params['task'] = task
        model_ok, train_ok, preprocesor_ok = mz.auto.prepare(
            model=model,
            data_pack=train_data_pack,
            verbose=0
        )
        print('model name: ', model_ok.params['name'])
        test_ok = preprocesor_ok.transform(test_data_pack, verbose=0)

        callback = mz.engine.callbacks.EvaluateAllMetrics(
            model_ok,
            *test_ok.unpack(),
            batch_size=1024,
            verbose=0
        )
        history = model_ok.fit(*train_ok.unpack(), batch_size=32, epochs=3,
                               callbacks=[callback])
        results.append({'name': model_ok.params['name'], 'history': history})
        print(model_ok.evaluate(*test_ok.unpack()))

    return results


if __name__ == '__main__':

    sample_pack = load_data()
    train_data, test_data = train_test_split(sample_pack)
    evaluate_results = auto_results(train_data, test_data)
    plot_results(evaluate_results)
