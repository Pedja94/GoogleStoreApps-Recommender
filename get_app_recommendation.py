import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def getRecommendedApps(appname, xdf, app_names, model, recommend_apps=5):

    appID = app_names.index[app_names['App'] == appname].tolist()
    if (len(appID) != 1):
        print('Invalid app name. Exit.')
        return []

    _, neighbors = model.kneighbors(xdf.loc[appID], n_neighbors=recommend_apps+1)

    similar_apps = []
    for neighbor in neighbors[0][1:]:
        similar_apps.append(app_names.loc[neighbor][0])

    similarity = []
    for app in similar_apps:
        simAppID = app_names.index[app_names['App'] == app].tolist()

        sim = cosine_similarity(xdf.loc[appID],xdf.loc[simAppID]).flatten()[0]
        similarity.append(sim*100)

    sim_df = pd.DataFrame({'App':similar_apps, 'Similarity':similarity})
    sim_df.sort_values(by='Similarity', ascending=False)

    return sim_df
#end of getRecommendedApps function

def main():

    app_names_df = pd.read_csv('App_Names.csv')
    data_df =  pd.read_csv('Cleaned_Data.csv')
    raw_data_df = pd.read_csv('./Dataset/googleplaystore.csv')

    model = NearestNeighbors(metric='euclidean')
    model.fit(data_df)

    recommed_apps = input('Number of similar apps to display: ')

    app_name = input('Find similar apps for: ')
   
    while (app_name != 'exit' and app_name != 'Exit'):
        id = raw_data_df.index[raw_data_df['App'] == app_name].tolist()
        print(raw_data_df.loc[id])
        print()

        similar_apps = getRecommendedApps(app_name, data_df, app_names_df, model, int(recommed_apps))

        print('Similar apps for app ', app_name, ' are:')
        print()
        print(similar_apps)
        print()

        if (len(similar_apps) != 0):
            for _, row in similar_apps.iterrows():
                name = row['App']

                id = raw_data_df.index[raw_data_df['App'] == name].tolist()
                print(raw_data_df.loc[id])

        print()

        app_name = input('Find similar apps for: ')

if __name__ == "__main__":
    main()