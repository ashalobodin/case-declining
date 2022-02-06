import click

from case_declining.models.encodec import EncoDec
from case_declining.models.vanilla_ltsm import VanillaLTSM
from case_declining.utils.data_utils import DataConverter


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
@click.option('--vdp', default='datasets/validation_data.txt', type=str, help='Validation data path')
@click.option('--testdp', default='datasets/test_data.txt', type=str, help='Test data path')
@click.option('--ngram-factor1', default=3, type=int, help='Surname ngram-factor')
@click.option('--ngram-factor2', default=2, type=int, help='Name ngram-factor')
@click.option('--ngram-factor3', default=2, type=int, help='Second name ngram-factor')
@click.pass_context
def declining(ctx, type, name, tdp, vdp, testdp, ngram_factor1, ngram_factor2, ngram_factor3):
    ctx.ensure_object(dict)
    ctx.obj['tdp'] = tdp
    ctx.obj['vdp'] = vdp
    ctx.obj['testdp'] = testdp
    ctx.obj['model_type'] = type
    ctx.obj['model_name'] = name
    ctx.obj['data_converter'] = DataConverter(tdp, vdp, testdp,
                                              ngram_factor=(ngram_factor1, ngram_factor2, ngram_factor3))


@declining.command()
@click.option('--hidden-units', type=int, default=80, help='# hidden units')
@click.option('--activation', type=str, default='softmax', help='Activation function')
@click.option('--epochs', type=int, default=600, help='# of epochs')
@click.option('--dropout', default=0.1, help='Dropout')
@click.option('--verbose', default=2, help='Verbose level')
@click.pass_context
def train(ctx, hidden_units, activation, epochs, dropout, verbose):
    dc = ctx.obj['data_converter']
    train_x = dc.get_column_encoded_data(dc.train_data)
    train_y = dc.get_column_encoded_data(dc.train_data, dc.column_y)
    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=False,
                      hidden_units=hidden_units, activation=activation, dropout=dropout)
    if model:
        model.fit(train_x, train_y, epochs=epochs, verbose=verbose)
        model.save()


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
@click.option('--data-set', type=click.Choice(['validation', 'test']), default='validation',
              help='Data set for prediction')
@click.pass_context
def predict(ctx, data_set):
    dc = ctx.obj['data_converter']
    input_data = dc.val_data if data_set == 'validation' else dc.test_data

    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=True)
    if model:
        data = dc.get_column_encoded_data(input_data)
        res = [dc.one_hot_decode(p, inp) for p, inp in zip(model.predict(data), input_data[dc.COLUMN_X])]
        print(f'Predictive genitive case are:')
        for i in range(len(input_data)):
            print(input_data[dc.COLUMN_X].iloc[i], '->', res[i])


@declining.command()
@click.option('--data-set', type=click.Choice(['validation', 'test']), default='validation',
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
@click.option('--hidden-units', type=int, multiple=True, default=[60, 80, 100], help='# hidden units')
@click.option('--input-str', type=str, default='Шалободін Олександр Олександрович')
@click.option('--verbose', type=int, default=0)
@click.pass_context
def learning_curve_hidden_units(ctx, hidden_units, input_str, verbose):
    dc = ctx.obj['data_converter']
    train_x = dc.get_column_encoded_data(dc.train_data)
    train_y = dc.get_column_encoded_data(dc.train_data, dc.column_y)

    # other hyper-parameters are constants
    activation = 'softmax'
    epochs = 500

    res = []

    for h_u in hidden_units:
        model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=False,
                          hidden_units=h_u, activation=activation)
        model.fit(train_x, train_y, epochs=epochs, verbose=verbose)
        pred = ["".join(dc.one_hot_decode(p, input_str)).title()
                for p in model.predict(dc.get_string_encoded(input_str))][0]
        val_score = calc_bleu_score(model, dc)
        test_score = calc_bleu_score(model, dc, "test_data")

        res.append({
            'Hidden units No': h_u,
            'validation_score': val_score,
            'generalization_score': test_score,
            'pred': pred
        })

        del model

    for r in res:
        for k, v in r.items():
            print(k, v)
        print()


@declining.command()
@click.option('--hidden-units', type=int, default=60, help='# hidden units')
@click.option('--input-str', type=str, default='Шалободін Олександр Олександрович')
@click.option('--verbose', type=int, default=0)
@click.pass_context
def learning_curve_ngrams(ctx, hidden_units, input_str, verbose):
    activation = 'softmax'
    epochs = 350
    res = []
    # ngram_factors = [(4, 3, 3), (3, 3, 3), (4, 15, 15), (3, 15, 15)]
    ngram_factors = [(3, 2, 2)]
    for nf in ngram_factors:
        dc = DataConverter(ctx.obj['tdp'], ctx.obj['vdp'], ctx.obj['testdp'], nf)
        train_x = dc.get_column_encoded_data(dc.train_data)
        train_y = dc.get_column_encoded_data(dc.train_data, dc.column_y)

        model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=False,
                          hidden_units=hidden_units, activation=activation)
        model.fit(train_x, train_y, epochs=epochs, verbose=verbose)
        pred = ["".join(dc.one_hot_decode(p, input_str)).title()
                for p in model.predict(dc.get_string_encoded(input_str))][0]
        val_score = calc_bleu_score(model, dc)
        test_score = calc_bleu_score(model, dc, "test_data")

        res.append({
            'ngram-factors': nf,
            'validation_score': val_score,
            'generalization_score': test_score,
            'pred': pred
        })

        del model

    for r in res:
        for k, v in r.items():
            print(k, v)
        print()


@declining.command()
@click.option('--dropout', multiple=True, default=[0., 0.2, 0.4, 0.6, 0.8], help='# hidden units')
@click.option('--input-str', type=str, default='Шалободін Олександр Олександрович')
@click.option('--verbose', type=int, default=0)
@click.pass_context
def learning_curve_dropout(ctx, dropout, input_str, verbose):
    activation = 'softmax'
    epochs = 650
    hidden_units = 80
    ngram_factors = (3, 2, 2)
    dc = DataConverter(ctx.obj['tdp'], ctx.obj['vdp'], ctx.obj['testdp'], ngram_factors)
    train_x = dc.get_column_encoded_data(dc.train_data)
    train_y = dc.get_column_encoded_data(dc.train_data, dc.column_y)
    res = []
    for dp in dropout:

        model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=False,
                          hidden_units=hidden_units, activation=activation, dropout=dp)
        model.fit(train_x, train_y, epochs=epochs, verbose=verbose)
        pred = ["".join(dc.one_hot_decode(p, input_str)).title()
                for p in model.predict(dc.get_string_encoded(input_str))][0]
        val_score = calc_bleu_score(model, dc)
        test_score = calc_bleu_score(model, dc, "test_data")

        res.append({
            'dropout': dp,
            'validation_score': val_score,
            'generalization_score': test_score,
            'pred': pred
        })

        del model

    for r in res:
        for k, v in r.items():
            print(k, v)
        print()


if __name__ == '__main__':
    declining()
