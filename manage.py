import json
from collections import namedtuple

import click
import numpy as np
import matplotlib.pyplot as plt

from case_declining.models.encodec import EncoDec
from case_declining.models.vanilla_ltsm import VanillaLTSM
from case_declining.utils.data_utils import DataConverter

from keras.optimizers import adam_v2


plot_type_index_map = {'loss': 0, 'accuracy': 1, 'val_loss': 2, 'val_accuracy': 3}
index_plot_type_map = {0: 'loss', 1: 'accuracy', 2: 'val_loss', 3: 'val_accuracy'}

TrainResult = namedtuple('TrainResult',
                         ['Model_description', 'Train_accuracy', 'Validation_accuracy', 'BLEU_score'])


def get_model(model_type, model_name, data_input_shape, load=False, **kwargs):
    if model_type == 'vanilla':
        model = VanillaLTSM(model_name, data_input_shape, load=load)
    elif model_type == 'encodec':
        model = EncoDec(model_name, data_input_shape, load=load)
    else:
        return None

    if not load:
        # kwargs should include hidden_units, activation
        model.compile(**kwargs)
    return model


def calc_bleu_score(model, dc, ds='val_data'):
    df = getattr(dc, ds)
    data = dc.get_column_encoded_data(df)
    res = [dc.one_hot_decode(p, inp) for p, inp in zip(model.predict(data), df[dc.COLUMN_X])]
    return model.bleu_score(df[dc.COLUMN_Y].tolist(), res)


@click.group()
@click.option('--type', type=click.Choice(['encodec', 'vanilla']), default='vanilla', help='Model type')
@click.option('--name', type=str, help='Model name')
@click.option('--tdp', default='datasets/train_data.txt', type=str, help='Train data path')
@click.option('--testdp', default='datasets/test_data.txt', type=str, help='Test data path')
@click.option('--ngram-factor1', default=3, type=int, help='Surname ngram-factor')
@click.option('--ngram-factor2', default=2, type=int, help='Name ngram-factor')
@click.option('--ngram-factor3', default=2, type=int, help='Second name ngram-factor')
@click.option('--load-tk', is_flag=True)
@click.pass_context
def declining(ctx, type, name, tdp, testdp, ngram_factor1, ngram_factor2, ngram_factor3, load_tk):
    ctx.ensure_object(dict)
    ctx.obj['tdp'] = tdp
    ctx.obj['testdp'] = testdp
    ctx.obj['model_type'] = type
    ctx.obj['model_name'] = name
    ctx.obj['data_converter'] = DataConverter(tdp, testdp, ngram_factor=(ngram_factor1, ngram_factor2, ngram_factor3))


@declining.command()
@click.option('--hidden-units', type=int, default=100, help='# hidden units')
@click.option('--activation', type=str, default='softmax', help='Activation function')
@click.option('--epochs', type=int, default=100, help='# of epochs')
@click.option('--dropout', default=0.1, help='Dropout')
@click.option('--validation-split', default=0.2)
@click.option('--learning-rate', default=1e-3)
@click.option('--verbose', default=2, help='Verbose level')
@click.option('--attempts', default=3)
@click.pass_context
def train(ctx, hidden_units, activation, epochs, dropout, validation_split, learning_rate, verbose, attempts):
    dc = ctx.obj['data_converter']
    train_x = dc.get_column_encoded_data(dc.train_data)
    train_y = dc.get_column_encoded_data(dc.train_data, dc.column_y)
    optimizer = adam_v2.Adam(learning_rate=learning_rate)
    max_test_score = 0
    max_history = None
    figure, axis = plt.subplots()
    for a in range(attempts):
        model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=False,
                          hidden_units=hidden_units, activation=activation, dropout=dropout, optimizer=optimizer)
        if model:
            history = model.fit(train_x, train_y, validation_split, epochs=epochs, verbose=verbose)
            test_score = calc_bleu_score(model, dc, "test_data")
            print(f'Attempt {a}: test score {test_score}')
            if test_score > max_test_score:
                max_test_score = test_score
                model.save()
                with open(f"{model.SAVED_MODELS_DIR}/{ctx.obj['model_name']}.json", 'w') as f:
                    json.dump(dc.tk.to_json(), f)
                max_history = history

    if max_history:
        print(f'Max test score {max_test_score}')
        axis.plot(list(max_history.values())[1], label=index_plot_type_map[1])
        axis.plot(list(max_history.values())[3], label=index_plot_type_map[3])
        axis.legend()
        plt.show()


@declining.command()
@click.pass_context
@click.option('--input-str', type=str, default='Шалободін Олександр Олександрович', help='Surname Name SecondName')
def predict_one(ctx, input_str):
    dc = ctx.obj['data_converter']
    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=True)
    if model:
        res = dc.one_hot_decode(model.predict(dc.get_string_encoded(input_str))[0], input_str)
        print(f'Predictive genitive case is `{res}`')


@declining.command()
@click.option('--data-set', type=click.Choice(['validation', 'test']), default='test',
              help='Data set for prediction')
@click.option('--diff', is_flag=True)
@click.pass_context
def predict(ctx, data_set, diff):
    dc = ctx.obj['data_converter']
    input_data = dc.val_data if data_set == 'validation' else dc.test_data

    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=True)
    if model:
        data = dc.get_column_encoded_data(input_data)
        res = [dc.one_hot_decode(p, inp) for p, inp in zip(model.predict(data), input_data[dc.COLUMN_X])]
        print(f'Predictive genitive case are:')
        # TODO: vectorize
        for i in range(len(input_data)):
            y = input_data[dc.COLUMN_Y].iloc[i]
            if not diff or y != res[i]:
                print(input_data[dc.COLUMN_X].iloc[i], '->', res[i])


@declining.command()
@click.option('--data-set', type=click.Choice(['validation', 'test']), default='test',
              help='Data set for scoring')
@click.pass_context
def bleu_score(ctx, data_set):
    dc = ctx.obj['data_converter']

    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=True)
    if model:
        score = calc_bleu_score(model, dc, 'val_data' if data_set == 'validation' else 'test_data')
        print(f'Bleu score for {data_set} dataset is {score}')


# ================== Learning curves ==================
@declining.command()
@click.option('--input-str', type=str, default='Шалободін Олександр Олександрович')
@click.option('--validation-split', default=0.2)
@click.option('--verbose', type=int, default=0)
@click.pass_context
def learning_curve_ngrams(ctx, input_str, validation_split, verbose):
    hidden_units = 80
    activation = 'softmax'
    epochs = 350
    ngram_factors = [(4, 15, 15), (4, 3, 3), (3, 15, 15), (3, 3, 3), (3, 2, 2)]
    for nf in ngram_factors:
        dc = DataConverter(ctx.obj['tdp'], ctx.obj['testdp'], nf)
        train_x = dc.get_column_encoded_data(dc.train_data)
        train_y = dc.get_column_encoded_data(dc.train_data, dc.column_y)

        model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=False,
                          hidden_units=hidden_units, activation=activation)
        model.fit(train_x, train_y, validation_split, epochs=epochs, verbose=verbose)
        # pred = ["".join(dc.one_hot_decode(p, input_str)).title()
        #         for p in model.predict(dc.get_string_encoded(input_str))][0]
        val_score = calc_bleu_score(model, dc)
        test_score = calc_bleu_score(model, dc, "test_data")

        print('ngram-factors', nf)
        print('validation_score', val_score)
        print('generalization_score', test_score)
        print()

        del model


@declining.command()
@click.pass_context
@click.option('--dropout', multiple=True, default=[0.05, 0.1, 0.15, 0.2])
@click.option('--unit', multiple=True, default=[80, 100, 120, 150])
@click.option('--learning-rate', multiple=True, default=[1e-3, 2e-3, 5e-3])
@click.option('--plot', type=click.Choice(['loss', 'accuracy', 'val_loss', 'val_accuracy']),
              multiple=True, default=['accuracy', 'val_accuracy'])
@click.option('--validation-split', default=0.25)
@click.option('--filename')
@click.option('--val-score-threshold', default=0.97)
@click.option('--attempts', default=3)
def learning_curve_all(ctx, dropout, unit, learning_rate, plot, validation_split, filename, val_score_threshold, attempts):
    plot_indices = [plot_type_index_map[p] for p in plot]
    activation = 'softmax'
    epochs = 80
    ngram_factors = (3, 2, 2)
    print(f'======= Plottinng learning curves for dropouts {dropout}, hidden units {unit}, '
          f'ngram_factors {ngram_factors}, learning rates {learning_rate} epochs {epochs} =======')

    dc = DataConverter(ctx.obj['tdp'], ctx.obj['testdp'], ngram_factors)
    train_x = dc.get_column_encoded_data(dc.train_data)
    train_y = dc.get_column_encoded_data(dc.train_data, dc.column_y)

    results = []
    acc_figure, acc_axis = plt.subplots()
    for i, dp in enumerate(dropout):
        for j, h_u in enumerate(unit):
            for lr in learning_rate:
                optimizer = adam_v2.Adam(learning_rate=lr)
                prev_val_score = 0
                figure, axis = plt.subplots()
                for a in range(attempts):
                    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=False,
                                      hidden_units=h_u, activation=activation, dropout=dp, optimizer=optimizer)
                    history = model.fit(train_x, train_y, validation_split, epochs=epochs, verbose=0)

                    test_score = calc_bleu_score(model, dc, "test_data")
                    print(f'\tAttempt {a} for dropout {dp}, units {h_u}, learning rate {lr}: {test_score}')

                    val_score = history["val_accuracy"][-1]
                    acc_axis.plot(list(history.values())[1], label=f'acc {dp} {h_u} {lr} ({a})')
                    if val_score >= val_score_threshold and val_score > prev_val_score:
                        prev_val_score = val_score
                        for p in plot_indices:
                            axis.plot(list(history.values())[p], label=index_plot_type_map[p])
                        axis.legend()

                        results.append(
                            TrainResult(f'DP {dp}, units {h_u}', history["accuracy"][-1], val_score, test_score)
                        )
                        print(results[-1])
                        figure.savefig(f"learning_curves/Dropout_{dp}_units_{h_u}_attempt_{a}.png")

                        if filename:
                            model.save(f'{model.SAVED_MODELS_DIR}/{filename}__{dp}_{h_u}_{lr}.h5')
                    plt.close(figure)
                    del model

    acc_axis.legend()
    plt.show()

    if results and filename:
        with open(f'learning_curves/{filename}.json', 'w') as f:
            json.dump(results, f)


@declining.command()
@click.pass_context
def error_analysis(ctx):
    dc = ctx.obj['data_converter']
    print(dc.tk.word_index.keys())
    input_data = dc.test_data

    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=True)
    if model:
        data = dc.get_column_encoded_data(input_data)
        dc.test_data['res'] = np.array([dc.one_hot_decode(p, inp) for p, inp in zip(model.predict(data), input_data[dc.COLUMN_X])])

        res = dc.test_data['res'].str.split(' ', expand=True)
        Y = dc.test_data['Y'].str.split(' ', expand=True)
        print('Surnames\n', Y[0].compare(res[0]))
        print('Names\n', Y[1].compare(res[1]))
        print('Second names\n', Y[2].compare(res[2]))
        # print(Y.compare(res).describe())


if __name__ == '__main__':
    declining()
