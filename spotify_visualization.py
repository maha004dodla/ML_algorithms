#importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data extraction
indexes=['track_name', 'artist(s)_name', 'artist_count', 'released_year',
       'released_month', 'released_day', 'in_spotify_playlists',
       'in_spotify_charts', 'streams', 'in_apple_playlists', 'in_apple_charts',
       'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts', 'bpm',
       'mode', 'key', 'danceability_%', 'valence_per', 'energy_perc',
       'acousticness_perc', 'instrumentalness_perc', 'liveness_perc',
       'speechiness_perc','count']
data=pd.read_csv(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\spotify-2023.csv",encoding='unicode_escape',names=indexes)
print(data.head())


#data understanding
print(data.shape)
print(data.columns)
print(data.index)

data.describe()


#data preprocessing
print(data.info())

data.isna().sum()
data.drop(data.head(1).index,inplace = True)
data.drop(data.tail(1).index,inplace = True)
data.isna().sum()


data['in_shazam_charts'].isna().sum()
print(data['in_shazam_charts'].value_counts())
data['in_shazam_charts'].fillna('0',inplace=True)
data['in_shazam_charts'].isna().sum()


data['mode'].isna().sum()
print(data['mode'].value_counts())
data['mode'].fillna('Major',inplace=True)
data['mode'].isna().sum()

data['key'].isna().sum()
print(data['key'].value_counts())
data['key'].fillna('C#',inplace=True)
data['key'].isna().sum()


data['count'].fillna('1', inplace = True)
data['count'].isna().sum()

data.isna().sum()


df=pd.DataFrame(data)
#data visualization


df['streams'] = pd.to_numeric(df['streams'], errors='coerce', downcast='integer').astype('Int64')
df['streams_in_millions']=df['streams']/1000000
df['streams_in_millions'].fillna(1.0,inplace=True)
df['detailed_name']=df['track_name'].astype(str) +"("+ df["artist(s)_name"]+")"


#plot for top 10 most streaming songs with artist name
fig,ax=plt.subplots(figsize=(25,8))
plt.xticks(rotation=45, ha="right")
sns.barplot(data=df.sort_values('streams_in_millions', ascending=False).head(10),x='detailed_name',y='streams_in_millions')
plt.xlabel('Song Name')
plt.ylabel('Stream in million')
plt.title('Top 10 Most-Streamed Songs')
plt.show()


#plot for top 20 artists with max no.of songs
plt.figure(figsize=(10,6))
sns.countplot(data=df,order=df['artist(s)_name'].value_counts(ascending=False).iloc[:20].index,y='artist(s)_name',palette='magma')
plt.xlabel('No. of songs')
plt.ylabel('Artist(s)')
plt.title('Top 20 artist with most number of songs')


#plot to print top 5 spotify playlist
df['in_spotify_playlists'] = pd.to_numeric(df['in_spotify_playlists'], errors='coerce', downcast='integer').astype('Int64')
#print(df['in_spotify_playlists'].dtype)
plt.figure(figsize=(10,7))
ax=sns.barplot(data=df,order=df.sort_values('in_spotify_playlists',ascending=False).track_name.iloc[:5],x='in_spotify_playlists',y='track_name',palette='RdBu')
plt.title('Top 5 songs in most of Spotify playlist')
plt.ylabel('Track')



#plot to describe top 15 apple playlist
df['in_apple_playlists'] = pd.to_numeric(df['in_apple_playlists'], errors='coerce', downcast='integer').astype('Int64')
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
ax=sns.barplot(x='in_apple_playlists',y='track_name',data=df,order=df.sort_values('in_apple_playlists',ascending=False).track_name.iloc[:15],palette='crest')
plt.title('Top 15 songs in most of Apple playlist')
plt.ylabel('Tracks')




#plot for type of divisions in mode column
music_grouped=df.groupby(by=['mode'])['streams_in_millions'].sum().reset_index()
colors = sns.color_palette('pastel')[0:10]
fig,ax=plt.subplots(figsize=(15,5))
plt.pie(music_grouped['streams_in_millions'], labels = music_grouped['mode'], colors = colors, autopct='%.0f%%')
plt.show()


#plot to find out distribution for all values
plt.figure(figsize=(10,3))
sns.histplot(data['danceability_%'],kde=True, color='purple')
plt.xlabel('Danceability (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability')
plt.show()


#plot to find out the count according to month
x = data.groupby('released_month')['released_month'].count().sort_values(ascending=False)
ax=x.plot(kind='bar',figsize=(10,5), grid=False, color='green')
ax.bar_label(ax.containers[0],label_type='center', color='white', weight='bold')
plt.title('Count of Tracks in 2023 by Month')
plt.xlabel('Number of Tracks')
plt.ylabel('Months')
plt.show()


#plot to show apple charts distribution
plt.figure(figsize=(16, 8))
sns.distplot(df.in_apple_charts,bins=20)



