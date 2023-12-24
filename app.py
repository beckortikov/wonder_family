import pandas as pd
import streamlit as st
st.set_page_config(layout='wide')



def main():
    # st.title("Приложение с вкладками")

    # Создаем боковую панель с выбором вкладки
    tab = st.sidebar.selectbox("Выберите задачу", ["Задача 1", "Задача 2"])

    # Отображаем содержимое в зависимости от выбранной вкладки
    if tab == "Задача 1":
        render_tab1()
    elif tab == "Задача 2":
        render_tab2()

def render_tab1():
    # st.header("Содержимое Вкладки 1")
    # st.write("Текст и компоненты для первой вкладки.")
    @st.cache_data
    def load_data(file):
        data = pd.read_excel(file, skiprows=3, sheet_name='Opt2')
        return data


    uploaded_file = st.sidebar.file_uploader('UPLOAD FILE OF TASK 1')


    if uploaded_file is None:
        st.info(" Uploade a file through config", icon='⛔️')
        st.stop()
    df = load_data(uploaded_file)


    df = df[['Продукт', 'Сроки','Продукт найден аналитиком', 'Составлена сводная таблица',
        'Добавлен в Gate 0 и approved', 'Передано в R&D и Marketing',
        'Запущен предв. поиск поставщика',
        'Проверка патентов, разрешительной документации', 'Ник проверил ROI',
        'Проведена встреча со специалистом / Изучение конкурентов',
        'Получение и финализация формулы от химика-технолога', 'UVP создано',
        'Отрисованы предв. дизайны', 'Получен фин себес от фабрики',
        'Фин модель утверждена', 'Презентация готова',
        'Готовы к ярмарке продуктов', 'Ярмарка продуктов',
        'Отправлен в галерею продуктов', 'Голден семпл заказан',
        'C онбординга до презентации продукта',
        'С презентации до утверждения партнёром', 'Семплы заказаны ',
        'Фабрика утверждена\n', 'Голден семпл утверждён', 'Старт производства',
        'Отправка на «Амазон» / в преп-центр', 'Появление стока на «Амазоне»']]

    df['Продукт'].fillna(method='ffill', inplace=True)

    df = df[(df["Сроки"] == "План даты:") | (df['Сроки']=='Факт, даты:')]

    df.replace('-', pd.NA, inplace=True)

    for columns in ['Продукт найден аналитиком', 'Составлена сводная таблица',
        'Добавлен в Gate 0 и approved', 'Передано в R&D и Marketing',
        'Запущен предв. поиск поставщика',
        'Проверка патентов, разрешительной документации', 'Ник проверил ROI',
        'Проведена встреча со специалистом / Изучение конкурентов',
        'Получение и финализация формулы от химика-технолога', 'UVP создано',
        'Отрисованы предв. дизайны', 'Получен фин себес от фабрики',
        'Фин модель утверждена', 'Презентация готова',
        'Готовы к ярмарке продуктов', 'Ярмарка продуктов',
        'Отправлен в галерею продуктов', 'Голден семпл заказан']:
        df[columns] = pd.to_datetime(df[columns])

    df['C онбординга до презентации продукта'] = pd.to_datetime(df['C онбординга до презентации продукта'])
    df['С презентации до утверждения партнёром'] = pd.to_datetime(df['С презентации до утверждения партнёром'])
    df['Семплы заказаны '] = pd.to_datetime(df['Семплы заказаны '].replace({'Еще нет': None, '3 ovt': '2023-10-03'}))

    df['Фабрика утверждена\n'] = pd.to_datetime(df['Фабрика утверждена\n'].replace({'12 sept': '2023-09-12', '10 sept': '2023-09-10'}))

    df['Старт производства'] = pd.to_datetime(df['Старт производства'].replace({'11 sept': '2023-09-11'}))
    df['Отправка на «Амазон» / в преп-центр'] = pd.to_datetime(df['Отправка на «Амазон» / в преп-центр'].replace({'10 maay': '2023-05-10', 'не отправлено': None}))
    df['Появление стока на «Амазоне»'] = pd.to_datetime(df['Появление стока на «Амазоне»'])
    df['Голден семпл утверждён'] = pd.to_datetime(df['Голден семпл утверждён'].replace({'В процессе': None, 'не было': None, 'Не было': None}))
    # df['Голден семпл утверждён'] = df['Голден семпл утверждён'].apply(lambda x: None if isinstance(x, str) else x)


    for columns in df.columns:
        try:
            df[columns] = (df.groupby('Продукт')[columns].diff().dt.days.fillna('Дата отсутствует'))
        except:
            if columns == 'Продукт' or columns == 'Сроки':
                continue
            df[columns] = df[columns].apply(lambda x: 'Не выполнено' if pd.isnull(x) else 'Выполнено')

    result_df = df[df['Сроки'] == 'Факт, даты:'][['Продукт', 'Продукт найден аналитиком',
        'Составлена сводная таблица', 'Добавлен в Gate 0 и approved',
        'Передано в R&D и Marketing', 'Запущен предв. поиск поставщика',
        'Проверка патентов, разрешительной документации', 'Ник проверил ROI',
        'Проведена встреча со специалистом / Изучение конкурентов',
        'Получение и финализация формулы от химика-технолога', 'UVP создано',
        'Отрисованы предв. дизайны', 'Получен фин себес от фабрики',
        'Фин модель утверждена', 'Презентация готова',
        'Готовы к ярмарке продуктов', 'Ярмарка продуктов',
        'Отправлен в галерею продуктов', 'Голден семпл заказан',
        'C онбординга до презентации продукта',
        'С презентации до утверждения партнёром', 'Семплы заказаны ',
        'Фабрика утверждена\n', 'Голден семпл утверждён', 'Старт производства',
        'Отправка на «Амазон» / в преп-центр', 'Появление стока на «Амазоне»']]



    # Streamlit App

    with st.sidebar:
    # Multiselect widget to select products
        selected_products = st.multiselect('Select Products', result_df['Продукт'])

    # Filter data based on selected products
    filtered_df = result_df[result_df['Продукт'].isin(selected_products)]
    filtered_df = result_df[result_df['Продукт'].isin(selected_products)]
    for col in filtered_df.columns:
        # Check if the column contains numeric values
        if pd.to_numeric(filtered_df[col], errors='coerce').notna().all():
            filtered_df[col] = filtered_df[col].astype(float)  # Convert to float
        else:
            if col == 'Продукт':
                continue
            filtered_df = filtered_df.drop(col, axis=1)  # Drop columns with string values

    def color_cell(value):
        if isinstance(value, (int, float)):
            color = 'green' if value <= 0 else 'red'
        else:
            color = 'white'
        return f'background-color: {color}; color: black;'

    # Применение стилизации к таблице
    styled_df = filtered_df.T.style.applymap(color_cell)
    # Display the processed DataFrame





    new_df = pd.DataFrame()
    for column in result_df.iloc[:,1:]:
        len_tovar = len(result_df[result_df[column]==0])

        new_df[column] = [f'{(len_tovar / len(result_df))*100:.2f}%']

    # Expander for "Процент совпадения по срокам (в этапах)"
    with st.expander("Процент совпадения по срокам (в этапах)"):
        # st.write(new_df.iloc[:, :5])
        # st.write(new_df.iloc[:, 5:9])
        # st.write(new_df.iloc[:, 9:15])
        # st.write(new_df.iloc[:, 15:20])
        # st.write(new_df.iloc[:, 20:26])
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        for i, j in zip(new_df.iloc[:, :6].columns, (col1, col2, col3, col4, col5, col6)):
            j.metric(i, new_df.iloc[:, :6][i][0])

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        for i, j in zip(new_df.iloc[:, 6:12].columns, (col1, col2, col3, col4, col5, col6)):
            j.metric(i, new_df.iloc[:, 6:12][i][0])

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        for i, j in zip(new_df.iloc[:, 12:18].columns, (col1, col2, col3, col4, col5, col6)):
            j.metric(i, new_df.iloc[:, 12:18][i][0])

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        for i, j in zip(new_df.iloc[:, 18:24].columns, (col1, col2, col3, col4, col5, col6)):
            j.metric(i, new_df.iloc[:, 18:24][i][0])

        col1, col2 = st.columns(2)
        for i, j in zip(new_df.iloc[:, 24:].columns, (col1, col2)):
            j.metric(i, new_df.iloc[:, 24:][i][0])

    # Expander for "Разница плана и факта по дням"
    with st.expander("Разница плана и факта по дням (по продуктам) Выберите слева название продукта" ):
        st.dataframe(styled_df, use_container_width=True)

def render_tab2():
    uploaded_file2 = st.sidebar.file_uploader('UPLOAD FILE OF TASK 2')

    if uploaded_file2 is None:
        st.info(" Uploade a file through config", icon='⛔️')
        st.stop()
    @st.cache_data
    def load_data2(file):
        # products = pd.read_excel('Copy of Time to 50k для ТЗ.xlsx', sheet_name='автомат', skiprows=1)
        plan = pd.read_excel(file, sheet_name='fromFin', header=None, names=['date', 'product','total_amount', 'total_revenue'])
        fact = pd.read_excel(file, sheet_name='fromDB')
        return plan, fact

    plan, fact = load_data2(uploaded_file2)

    plan['total_revenue'] = plan[['total_revenue']][(plan[['total_revenue']].applymap(lambda x: isinstance(x, (int, float))))].values
    plan.total_revenue =round(plan.total_revenue.astype(float),2)
    plan.date = pd.to_datetime(plan.date)
    fact.month_date = pd.to_datetime(fact['month_date'])

    product = {'Microneedling Pen': ['B0CDNRQHNF', 'B0CFD9RXWW'],
            'Hand Mask 1 pack': ['B0CDCCLZ88','B0CF5PF6SR','B0CF5T63MQ','B08GYQFJ2F', 'B07JQLJS9M',
                                    'B0CF5XMSZP', 'B09YJ7KGRV', 'B07MGW8VWT', 'B0CF5VYVVZ', 'B0CF5YQPPY'],
            'Hand Mask 6 pack': ['B0CF5XX53L', 'B0CDCCGS46'],
            'Hand Mask 12 pack': ['B0CDCBZMFC', 'B08JXJNQTD'],
            'Hand Masks - 24 pack': ['B08JX79PNX'],
            'Foot M 3P': ['B0CFYXLJWM','B0CFZK9JJR','B0CFZLXXP9'],
            'Foot M 6P': ['B0CFZG51XB','B0CFZD6GHX', 'B0CFZHBFXB'],
            'Charcoal Peel Off Mask': ['B0CC9N5PSK', 'B0CC72XW21', 'B0CCB1P4B3', 'B0CBZKK4DJ'],
            'Moisturizing Serum': ['B0CB4SHY82', 'B0CFYQSW6P', 'B0CHFFQVY9', 'B0CHFGPVZX', 'B0CHFCP5YX'],
            'Curler': ['B0CH8TW7QP','B0CB1SNYB3', 'B0CH8SYF9Z'],
            'Green Stick 1 Pack': ['B0CFRJF3QR', 'B0CB4KFMHD', 'B0CFRDZX2C'],
            'Green Stick 2 Pack': ['B0CB4W6BN1'],
            'Eye P 15P':['B07JPHMY39', 'B0CFZQGBNV'],
            'Eye P 30P': ['B0CFZLVZ2S','B0CFZNQTN2','B0CFZLP75W','B0CFZ9W8QF'],
            'Bubble Clay Face Mask': ['B0CCVP9P52', 'B0C9TP6764', 'B0BSXKMVMT', 'B0CCVRPJLT', 'B0CB4TFCVB', 'B0CCVVP2TZ'],
            'Gold M 5P': ['B0CFYM3SKP','B0CFY9FH4V', 'B0CFYN8P88'],
            'Gold M 10P': ['B0CFYDDQZL', 'B0CFYMS147'],
            'Gold M 15P': ['B0CFYDSK7Q'],
            'Shower Steamers': ['B0CGRXK1B1', 'B0CJYFX7P4', 'B0CJYF81V6', 'B0CJYFR5J4']

            }

    product = pd.concat({key: pd.Series(value) for key, value in product.items()}, axis=1)

    f = fact.copy()

    f = f.reset_index(drop=True)
    l = pd.DataFrame()
    for k,i in enumerate(f['product_name']):
        for j in product:
            if i in product[j].to_list():
                l.loc[k, 'name'] = j

    merged_df = pd.merge(f, l, left_index=True, right_index=True, how='inner')
    merged_df.rename(columns={'month_date': 'date', 'name': 'product'}, inplace=True)
    merged_df = merged_df.groupby(['date','product'], as_index=False).sum().sort_values('total_revenue', ascending=False)
    za = pd.merge(merged_df, plan, on=['date', 'product'], how='inner')
    za = za.groupby(['date','product'], as_index=False).sum()
    za = za.fillna(0)
    za['difference']=za['total_revenue_x'] - za['total_revenue_y']

    def calculate_match_percentage(difference, total_revenue):
        # Проверка деления на ноль
        if total_revenue != 0:
            match_percentage = (difference / total_revenue) * 100
        else:
            match_percentage = 0

        return match_percentage

    # Применение функции к DataFrame
    za['match_percentage'] = za.apply(lambda row: calculate_match_percentage(row['difference'], row['total_revenue_y']), axis=1)

    product_metric = za['match_percentage'].sum()/80
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Средний процент совпадения по выруче", f'{product_metric:.2f}')


if __name__ == "__main__":
    main()
