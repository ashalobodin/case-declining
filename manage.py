import click

from case_declining.models.encodec import EncoDec
from case_declining.models.vanilla_ltsm import VanillaLTSM
from case_declining.utils.data_utils import DataConverter


@click.group()
@click.option('--type', type=click.Choice(['encodec', 'vanilla']), default='vanilla', help='Model type')
@click.option('--name', type=str, help='Model name')
@click.option('--tdp', default='datasets/train_data.txt', type=str, help='Train data path')
@click.option('--vdp', default='datasets/validation_data.txt', type=str, help='Validation data path')
@click.option('--testdp', default='datasets/test_data.txt', type=str, help='Test data path')
@click.pass_context
def declining(ctx, type, name, tdp, vdp, testdp):
    ctx.ensure_object(dict)
    ctx.obj['model_type'] = type
    ctx.obj['model_name'] = name
    ctx.obj['data_converter'] = DataConverter(tdp, vdp, testdp)


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


@declining.command()
@click.option('--hidden-units', type=int, default=40, help='# hidden units')
@click.option('--activation', type=str, default='softmax', help='Activation function')
@click.option('--epochs', type=int, default=500, help='# of epochs')
@click.pass_context
def train(ctx, hidden_units, activation, epochs):
    dc = ctx.obj['data_converter']
    train_x = dc.get_column_encoded_data(dc.train_data)
    train_y = dc.get_column_encoded_data(dc.train_data, dc.COLUMN_Y)

    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=False,
                      hidden_units=hidden_units, activation=activation)
    if model:
        model.fit(train_x, train_y, epochs=epochs)
        model.save()


@declining.command()
@click.pass_context
@click.option('--pib', type=str, help='Surname Name SecondName')
def predict_one(ctx, pib):
    dc = ctx.obj['data_converter']

    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=True)
    if model:
        P = model.predict(dc.get_string_encoded(pib))
        print(f'Predictive genitive case is `{["".join(dc.one_hot_decode(p)).title() for p in P][0]}`')


@declining.command()
@click.option('--data-set', type=click.Choice(['validation', 'test']), default='validation',
              help='Data set for prediction')
@click.pass_context
def predict(ctx, data_set):
    dc = ctx.obj['data_converter']

    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=True)
    if model:
        data = dc.get_column_encoded_data(dc.val_data if data_set == 'validation' else dc.test_data)
        P = model.predict(data)
        print(f'Predictive genitive case are:\n{["".join(dc.one_hot_decode(p)).title() for p in P]}')


@declining.command()
@click.option('--data-set', type=click.Choice(['validation', 'test']), default='validation',
              help='Data set for scoring')
@click.pass_context
def bleu_score(ctx, data_set):
    dc = ctx.obj['data_converter']

    model = get_model(ctx.obj['model_type'], ctx.obj['model_name'], dc.input_shape, load=True)
    if model:
        df = dc.val_data if data_set == 'validation' else dc.test_data
        data = dc.get_column_encoded_data(df)
        P = model.predict(data)

        print(f'Bleu score for {data_set} dataset is '
              f'{model.bleu_score(df[dc.COLUMN_Y].tolist(), ["".join(dc.one_hot_decode(p)).title() for p in P])}')


if __name__ == '__main__':
    declining()
