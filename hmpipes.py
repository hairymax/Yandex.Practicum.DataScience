import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

# Предобработка данных
def fillna_with_group_agg(df: pd.DataFrame, target_col: str, group_col: str, func='mean'):
    ''' Заполнение пропусков в столбце аггрегирующей функцией по сгруппированным значениям другого столбца

    Параметры:
    ---
    `df` : pd.DataFrame
    `target_col` : str - столбец, в котором заполняются пропуски
    `group_col` : str - столбец, по которому группируются значения
    `func` : str / numpy функция / lambda функция 
    '''
    df_new = df.copy()
    before = df[target_col].isna().sum()
    df_new.loc[df[target_col].isna(), target_col] = (
        df_new.loc[df[target_col].isna(), group_col].map(
            df_new.groupby(group_col)[target_col].agg(func))
    )
    print('{} out of {} values are filled with {} of "{}" among "{}"'
          .format(before-df_new[target_col].isna().sum(), before, func, target_col, group_col))
    return df_new

# Каэффициент корелляции Cramers-V для двух категориальных признаков
def cramers_corrected_stat(a, b):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = ss.contingency.crosstab(a, b)[1]
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

# ----------------------------------------------------------------------------------
# Визуализация данных
# ----------------------------------------------------------------------------------
def dataframe_info(df):
    ''' Вывод информации о датафрейме `df`. Функционал схож с методом `pd.DataFrame.info()`

    Дополнительно: вывод процента заполненных (non-null) значений, количестве дублей
    '''
    info = (pd.DataFrame({
        'notNA': df.count(), 'notNA, %': df.count()/df.shape[0]*100,
        'dtype': df.dtypes})
        .style.bar(subset='notNA, %', vmin=0, color='lightblue')
        .format("{:.2f}", subset=['notNA, %'])
    )
    print('DataFrame shape  : {} rows, {} columns'.format(
        df.shape[0], df.shape[1]))
    print('Memory usage     : {:.2f} MB'.format(
        df.memory_usage().sum()/1024/1024))
    print('Duplicates count :', df.duplicated().sum())
    return info


def boxplots(df: pd.DataFrame, cols: list, title='Boxplots', figsize=(12, 7)):
    ''' Диаграммы размаха для колонок cols из датасета df

    Параметры:
    ---
    `df` : :pd.DataFrame
    `cols` : list - названия столбцов, для которых строятся диаграммы размаха
    `title` : str - название графика
    '''
    # строим диаграммы размаха
    # инициализируем фигуру для диаграмм размаха, задаём для каждого столбца из списка свой subplot
    _, axs = plt.subplots(1, len(cols), figsize=figsize)
    for i, ax in enumerate(axs.flat):
        # строим диаграмму размаха, отфильтровав пропуски
        ax.boxplot(df[df[cols[i]].notna()][cols[i]])
        ax.set_title(cols[i], fontsize=14)  # заголовок
        ax.grid(visible=True)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xticks([], [])
        plt.suptitle(title, fontsize=14)
    plt.show()


def plot_hist_and_box(series_list: list, bins=100, figsize=(12, 8), legend=None,
                      xlim=None, ylim=None,
                      xlabel=None, ylabel='Values counts'):
    '''
    Функция построения гистограмм и диаграмм размаха для pd.Series из series
    
    Параметры 
    ---
    `bins` : int - задаёт число корзин для гистограммы, по умолчанию 100
    `xlabel, ylabel` : string - подписи к осям, если необходимо указать свои
    `xlim, ylim, figsize` : параметры matplotlib 
    '''

    if not (type(series_list) == list or type(series_list) == tuple):
        print("series should be list/tuple of pd.Series")
        return

    height_ratios = [.1]*len(series_list)
    height_ratios.append(1-sum(height_ratios))
    _, axs = plt.subplots(len(series_list)+1, sharex=True, figsize=figsize,
                          gridspec_kw={"height_ratios": height_ratios})
    cmap = plt.get_cmap('tab10')#'brg')
    #color_cnt = len(series_list)-1 if len(series_list) > 1 else 1
    names = ', '.join(set([s.name for s in series_list]))

    axs[0].set(title="Distribution of %s" % names)
    for i in range(len(series_list)):
        # диаграмма размаха
        sns.boxplot(x=series_list[i], ax=axs[i], color=cmap(i/10))
        axs[i].set(xlabel=series_list[i].name)
        # гистограмма с функцией распределения
        sns.histplot(series_list[i], ax=axs[-1], bins=bins, kde=True,
                     color=cmap(i/10))

    if not xlabel:
        xlabel = names
    axs[-1].set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    axs[-1].grid(visible=True)

    if legend:
        plt.legend(legend, fontsize='large')

    plt.show()


def plot_value_counts(series, n_values=25, fillna='NONE', figwidth=15, bar_thickness=0.5, verbose=True, show_percents=False):
    ''' Визуализация количества встречающихся значений в pd.Series

    Параметры
    ---
    `series` : pd.Series
    `column` : str - название столбца
    `n_values` : int - максимальное количество значений для отображения на диаграмме
    `fillna` : Any - значение, которым необходимо заполнить пропуски
    `verbose`: bool - показывать ли уникальные значения
    `show_percents`: bool - показывать долю значений в процентах
    '''
    _ = series.dropna().unique()
    if verbose:
        print('`{}`, {} unique values: \n{}'.format(series.name, len(_), sorted(_)))

    val_counts = series.fillna(fillna).value_counts()
    bar_values = val_counts.values[:n_values]
    bar_labels = val_counts.index[:n_values].astype('str')
    plt.figure(figsize=(figwidth, bar_thickness * min(len(val_counts), n_values)))
    ax = sns.barplot(x=bar_values, y=bar_labels)
    ax.set(title='"{}" value counts ({} / {})'
           .format(series.name, len(bar_labels), val_counts.shape[0]),
           #xlim=[0, 1.07*bar_values.max()]
           )
    if show_percents:
        labels = [f'{w/val_counts.values.sum()*100:0.1f}%' 
                    if (w := v.get_width()) > 0 else '' for v in ax.containers[0]]
    else:
        labels = bar_values
    plt.bar_label(ax.containers[0], labels=labels, label_type='center')
    for i in range(len(bar_labels)):
        if bar_labels[i] == fillna:
            ax.patches[i].set_color('black')
    plt.show()


# ----------------------------------------------------------------------------------
# Визуализация временных рядов
# ----------------------------------------------------------------------------------
def plot_time_series(series, roll_size=24, figsize=(16.5, 5)):
    """
    Функция построения графика временного ряда со скользящим средним и 
    стандартным отклонением по окну заданного размера
    
    Принимает
    ---
    `series` : pd.Series - временной ряд  
    `roll_size` : int - размер окна
    """
    pd.DataFrame({series.name: series,
                 'Rolling mean': series.rolling(roll_size).mean(),
                  'Rolling std': series.rolling(roll_size).std()}
                 ).plot(figsize=figsize)
    plt.grid()
    plt.xlabel("Time interval: %s - %s" %
               (series.index.min(), series.index.max()))
    plt.ylabel(series.name)
    plt.title("Time series of '%s' (rolling size = %i hours)" %
              (series.name, roll_size))
    plt.show()


def plot_average_by_period(series, x, hue=None, tot=False, xstep=1, figsize=(16.5, 5)):
    """Функция построения зависимости значений в `series` от временного промежутка `x` 
    с группировкой по `hue`

    Параметры
    ---
    `series` : pd.Series - временной ряд
    `x` : str, период по которому усреднять (`hour`/`day`/`weekday`/`month`/`year`)
    `hue` : str, период по которому группировать (`hour`/`day`/`weekday`/`month`/`year`)
    `tot` : bool - флаг строить ли усреднение по всем (несгруппированным) значениям
    `xstep` : int - шаг делений по оси x
    """
    _, ax = plt.subplots(figsize=figsize)
    try:
        x_ = getattr(series.index, x)
    except:
        print("Can't find '{}'".format(x))
        return
    title = "Dependence of '%s' on '%s'" % (series.name, x)

    if hue is not None:
        hue_ = getattr(series.index, hue)
        palette = sns.color_palette("husl", len(hue_.unique()))
        sns.lineplot(x=x_, y=series.values, hue=hue_, palette=palette)
        ax.legend(title=hue)
        title += " for different '%s'" % hue

    if hue is None or tot:
        sns.lineplot(x=x_, y=series.values, color='black')

    title += " on period %s - %s" % (series.index.min(), series.index.max())

    ax.set(xlabel=x, xlim=(x_.min(), x_.max()),
           xticks=(np.arange(x_.min(), x_.max()+1, xstep)),
           title=title)
    plt.grid()
    plt.show()


def plot_seasonal_decompose(decomposition, out='otsr', axsize=(16.5, 3)):
    """ Функция для более удобного вывода графификов из результата работы функции
    `tsa.seasonal.seasonal_decompose()`

    Принимает
    --- 
    `decomposition` : объект структуры DecomposeResult
    `out` : строка с ключами необходимых для вывода графиков 
            (`o` - observed, `t` - trend, `s` - seasonal, `r` - resid)
    """
    if len(out) < 1:
        return print('Nothing to show')

    _, axs = plt.subplots(len(out), figsize=(axsize[0], axsize[1]*len(out)))
    if len(out) == 1:
        axs.set_title("Seasonal decomposition")
        axs = {out: axs}
    else:
        axs[0].set_title("Seasonal decomposition")
        axs = {p: axs[i] for p, i in zip(out, range(len(out)))}

    if 'o' in out:
        decomposition.observed.plot(ax=axs['o'], color='darkblue', grid=True,
                                    ylabel='Observed', xlabel='')
    if 't' in out:
        decomposition.trend.plot(ax=axs['t'],  color='darkorange', grid=True,
                                 ylabel='Trend', xlabel='')
    if 's' in out:
        decomposition.seasonal.plot(ax=axs['s'], color='darkgreen', grid=True,
                                    ylabel='Seasonal', xlabel='')
    if 'r' in out:
        decomposition.resid.plot(ax=axs['r'],  color='darkred', grid=True,
                                 ylabel='Residual', xlabel='')
    plt.show()


# ----------------------------------------------------------------------------------
# Немножко автоматизации для обучения моделей
# ----------------------------------------------------------------------------------

def best_cv_models(grid, count, score='score'):
    ''' Выводит таблицу с показателями моделей, показавших наилучшие значения метрики на кроссвалидации.

    Принимает  
        : `grid` - результат GridSearchCV после fit(), 
        : `count` - количество лучших моделей для вывода
        : `score` - метрика
    Возвращает : pd.DataFrame c параметрами моделей
    '''
    print('Estimator: {}'.format(grid.estimator))
    print('Tested {} models. Splits: {}'.format(
        len(grid.cv_results_['params']), grid.cv))
    print('Best score = {}\n'.format(grid.best_score_))

    results = {}
    for s in set(['score', score]):
        if 'rank_test_'+s in grid.cv_results_.keys():
            best_idx = grid.cv_results_['rank_test_'+s].argsort()[:count]
        if 'mean_test_'+s in grid.cv_results_.keys():
            results['test '+score] = grid.cv_results_['mean_test_'+s][best_idx]
        if 'mean_train_'+s in grid.cv_results_.keys():
            results['train '+score] = grid.cv_results_['mean_train_'+s][best_idx]
    results['fit time, s'] = grid.cv_results_['mean_fit_time'][best_idx]
    results['score time, s'] = grid.cv_results_['mean_score_time'][best_idx]

    return pd.DataFrame(results).join(
        pd.DataFrame([grid.cv_results_['params'][i] for i in best_idx]))


def test_model(model, X_train, X_test, y_train, y_test, score_func=None):
    ''' 
    - Обучение модели `model` на выборках `X_train`, `y_train`
    - Предсказание обученной модели на наборе `X_test`
    - Вычисление метрики `score_func` на полученных предсказаниях и наборе `y_test`
    
    `score_func` : sklearn.metrics  

    Возвращает score, время обучения, время пердсказаний и сами предсказания в виде словаря
    '''
    # обучение
    t_beg = time.time()
    model.fit(X_train, y_train)
    time_fit = time.time() - t_beg
    # предсказания
    t_beg = time.time()
    y_pred = model.predict(X_test)
    time_predict = time.time() - t_beg
    # метрика
    score = score_func(y_test, y_pred)

    return {'score test': score,
            'fit time': time_fit,
            'predict time': time_predict,
            'predictions': y_pred}
