import pickle
import streamlit as st
import requests 
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_player import st_player
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
import hashlib
import sqlite3 
import ast
from PIL import Image

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


st.set_page_config(layout="wide")
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
""", unsafe_allow_html=True)

def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			pos_list.append(i)
			pos_list.append(res)

		elif res <= -0.1:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

def fetch_poster(movie_id):
	url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
	data = requests.get(url)
	data = data.json()
	poster_path = data['poster_path']
	full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
	return full_path

def fetch_details(movie_id):
	url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
	data = requests.get(url)
	data = data.json()
	return data

def fetch_vid(movie_id):
	url = "https://api.themoviedb.org/3/movie/{}/videos?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
	data = requests.get(url)
	data = data.json()
	key_value = data['results'][0]['key']
	#print(key_value,type(key_value))
	#print("https://www.youtube.com/watch?v={}".format(key_value))
	return key_value

def recommend(movie):
	#print("----------")
	#print(movie)
	#print("----------")
	movies = pickle.load(open('model/movie_list.pkl','rb'))
	similarity = pickle.load(open('model/similarity.pkl','rb'))
	index = movies[movies['title'] == movie].index[0]
	distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
	recommended_movie_names = []
	recommended_movie_posters = []
	main_movie_id = []
	k=0
	for i in distances[0:6]:
        # fetch the movie poster
		k = k+1
		movie_id = movies.iloc[i[0]].movie_id
		#print(movie_id)
		recommended_movie_posters.append(fetch_poster(movie_id))
		recommended_movie_names.append(movies.iloc[i[0]].title)
		if k==1:
			main_movie_id.append(movie_id)

        
	return recommended_movie_names,recommended_movie_posters,main_movie_id

def tags_genre(genre):
	st.markdown(f"""
		<span class="badge rounded-pill bg-primary btn-lg">{genre}</span>
	""",unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_data():
	df = pd.read_csv("filtered_track_df.csv")
	df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
	exploded_track_df = df.explode("genres")
	return exploded_track_df

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]
exploded_track_df = load_data()

#KNN Model
def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
	genre = genre.lower()
	genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
	genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]
	neigh = NearestNeighbors()#done till 23
	neigh.fit(genre_data[audio_feats].to_numpy())
	n_neighbors = neigh.kneighbors([test_feat],       n_neighbors=len(genre_data), return_distance=False)[0]
	uris = genre_data.iloc[n_neighbors]["uri"].tolist()
	audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
	return uris, audios



def main():
	page_bg_img = '''
	<style>
      .stApp {
  	background-image: url("https://payload.cargocollective.com/1/11/367710/13568488/MOVIECLASSICSerikweb_2500_800.jpg");
  	background-size: cover;
	}
	</style>'''

	# st.markdown(page_bg_img, unsafe_allow_html=True)
	menu_main = ["Home","Login","SignUp"]
	choice = st.sidebar.selectbox("Menu",menu_main)

	
	if choice == "Home":
		page_bg_img = '''
		<style>
      	.stApp {
  		background-image: url("https://payload.cargocollective.com/1/11/367710/13568488/MOVIECLASSICSerikweb_2500_800.jpg");
  		background-size: cover;
		}
		</style>'''

		# st.markdown(page_bg_img, unsafe_allow_html=True)
	
		# st.header("Welcome to Rec-Ent")
		st.markdown("<style>h1 {font-size: 70px;}<style/>",unsafe_allow_html=True)
		st.markdown("<h1>Welcome to Rec-Ent !!<h1/>",unsafe_allow_html=True)
		
		colum1, colum2 = st.columns((1,1.5))
		with colum1:
			# st.header(recommended_movie_names[0])
			image = Image.open('images\music.png')
			st.image(image,width=400)	
			#images\mov2.jfif				
		with colum2:
			st.header("")
			st.markdown("<style>p{font-size: 30px;}<style/>",unsafe_allow_html=True)
		
			# st.markdown('<style>p {}</style>', unsafe_allow_html=True)
			st.markdown('<p>Rec-Ent is a ML based recommendation system for entertainment.Rec-Ent stands for Recommend Entertainment.It recommends songs and movies to users.</p>', unsafe_allow_html=True)
			st.markdown('<br/>',unsafe_allow_html=True)
			st.markdown('<br/>',unsafe_allow_html=True)
			st.markdown('<br/>',unsafe_allow_html=True)

		co1,co2 = st.columns((1.5,1))	
		with co1:
			st.header("")
			st.markdown("<style>p{font-size: 30px;}<style/>",unsafe_allow_html=True)
			# st.markdown('<style>p {}</style>', unsafe_allow_html=True)
			st.markdown('<p>Rec-ent is a one stop solution for all your entertainment needs.We help you pick the right movie and right song  saving you a lot of time.</p>', unsafe_allow_html=True)
			st.markdown('<br/>',unsafe_allow_html=True)
			st.markdown('<br/>',unsafe_allow_html=True)
			st.markdown('<br/>',unsafe_allow_html=True)
			st.markdown('<br/>',unsafe_allow_html=True)
			st.markdown('<br/>',unsafe_allow_html=True)
		
		with co2:
			imae = Image.open('images\movie.jfif')
			st.image(imae,width=300)

		st.markdown('<br/>',unsafe_allow_html=True)
	
		components.html(
    """
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        
     <style>
            html {
            box-sizing: border-box;
            }

            *, *:before, *:after {
            box-sizing: inherit;
            }

            .column {
            float: left;
            width: 33.3%;
            margin-bottom: 16px;
            padding: 0 8px;
            }

            @media screen and (max-width: 650px) {
            .column {
                width: 80%;
                display: block;
            }
            }

            .card {
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            }

            .container {
            padding: 0 16px;
            }

            .container::after, .row::after {
            content: "";
            clear: both;
            display: table;
            }

            .title {
            color: grey;
            }

            .button {
            border: none;
            outline: 0;
            display: inline-block;
            padding: 8px;
            color: white;
            background-color: #000;
            text-align: center;
            cursor: pointer;
            width: 100%;
            }

            .button:hover {
            background-color: #555;
            }
			.center {
  			display: block;
  			margin-left: auto;
  			margin-right: auto;
  			width: 50%;
}
        </style>
    </head>
        <body>

            <h2>Meet The Team</h2>
            <div class="row">
            
            <div class="column">
                <div class="card">
                <img src="https://github.com/Ruchika-11/Air-Canvas/blob/main/images/Sadhana.jpeg?raw=true"alt="Sadhana"
                 style="width:85%" class="center"  width="100" height="200">
                <div class="container">
                    <h2>S Sadhana</h2>
                    <p class="title">Team Lead</p>
                    <p>18ETCS002102</p>
                    <p>Student</p>
					<p>sadhana1058@gmail.com</p>
                    <p><button class="button">Contact</button></p>
                </div>
                </div>
            </div>
            
            <div class="column">
                <div class="card">
                <img src="https://github.com/Ruchika-11/Air-Canvas/blob/main/images/WhatsApp%20Image%202022-01-10%20at%2010.52.38%20PM.jpeg?raw=true" 
                 width="100" height="200" alt="Sunidhi" style="width:85%" class="center" >
                <div class="container">
                    <h2>Sunidhi V</h2>
                    <p class="title">Team Member</p>
                    <p>18ETCS002125</p>
                    <p>Student</p>
                    <p>sunidhivenkatesh2000@gmail.com</p>
                    <p><button class="button">Contact</button></p>
                </div>
                </div>
            </div>

       

            </div>
           <div class="row">
                 <div class="column">
                <div class="card">
                <img src="https://github.com/Ruchika-11/Air-Canvas/blob/main/images/WhatsApp%20Image%202022-01-11%20at%205.25.10%20PM.jpeg?raw=true" 
                alt="Nithin " style="width:85%" class="center"  width="80" height="150">
                <div class="container">
                    <h2>Nithin Rao R</h2>
                    <p class="title">Mentor</p>
                    <p>Asst Professor</p>
                    <p>M.S. Ramaiah University</p>
                    <p>nithinrao@gmail.com</p>
                    <p><button class="button">Contact</button></p>
                </div>
                </div>
            </div>
           
           </div>

            </body>
    """,
        height=1200,
    )
       	
				
	
	elif choice == "Login":
		page_bg_img = '''
		<style>
      	.stApp {
  		background-image: url("https://payload.cargocollective.com/1/11/367710/13568488/MOVIECLASSICSerikweb_2500_800.jpg");
  		background-size: cover;
		}
		</style>'''

		# st.markdown(page_bg_img, unsafe_allow_html=True)
	
		# st.header("W
		# st.subheader("Rec-Ent Login")
		st.markdown("<style>h1 {font-size: 70px;}<style/>",unsafe_allow_html=True)
		st.markdown("<h1>Rec-Ent Login<h1/>",unsafe_allow_html=True)
		st.markdown("<style>p{font-size: 30px;}<style/>",unsafe_allow_html=True)
		# st.markdown('<p>Login to our Rec-Ent for more movie and song recommendations<p/>',unsafe_allow_html=True)

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				colu1,colu2 = st.columns((1,1))
				with colu1:
					st.header(" ")
				# st.success("Logged In as {}".format(username))
				with colu2:
					st.success("Welcome {}!!".format(username))

				menu = ["Movie Recommendation" ,"Genre Based Recommendation","Song Recommendation","Rate Rec-Ent"]
				choice = st.sidebar.selectbox("Menu",menu)

				movies = pd.read_csv(r'kaggle\input\tmdb_5000_movies.csv')
				credits = pd.read_csv(r'kaggle\input\tmdb_5000_credits.csv') 

				if choice == "Movie Recommendation":
					st.header('Movie Recommender System')
					moviespkl = pickle.load(open('model\movie_list.pkl','rb'))
					similarity = pickle.load(open('model\similarity.pkl','rb'))
					movie_list = moviespkl['title'].values
					selected_movie = st.selectbox(
						"Type or select a movie from the dropdown",
						movie_list
					)

					if st.button('More Details'):
						col_1, col_2 = st.columns((1,1.5))
						moviespkl = pickle.load(open('model\movie_list.pkl','rb'))
						movie_list = moviespkl['title'].values
						recommended_movie_names,recommended_movie_posters,main_mov_id = recommend(selected_movie)
						main_movie_data = fetch_details(main_mov_id[0])
						movie_video_key = fetch_vid(main_mov_id[0])
						genre_list=main_movie_data['genres']
						# print(type(genre_list),genre_list)
						all_genres = [d['name'] for d in genre_list]


						with col_1:
							st.image(recommended_movie_posters[0])
							
						with col_2:
							st.header(recommended_movie_names[0])
							st.subheader(main_movie_data['tagline'])
							st.write("Released On :{}".format(main_movie_data['release_date']))
							for current_genre in all_genres:
								tags_genre(current_genre)
							st.markdown("""<p><i class="fa fa-clock-o" aria-hidden="true">{}min</i></p>""".format(main_movie_data['runtime']),unsafe_allow_html=True)
							st.markdown(main_movie_data['overview'])
							st_player('https://www.youtube.com/watch?v={}'.format(movie_video_key))

						

					if st.button('Show Recommendation'):
						col_1, col_2 = st.columns((1,1.5))
						moviespkl = pickle.load(open('model\movie_list.pkl','rb'))
						movie_list = moviespkl['title'].values
						recommended_movie_names,recommended_movie_posters,main_mov_id = recommend(selected_movie)
						main_movie_data = fetch_details(main_mov_id[0])
						movie_video_key = fetch_vid(main_mov_id[0])
						genre_list=main_movie_data['genres']
						# print(type(genre_list),genre_list)
						all_genres = [d['name'] for d in genre_list]


						with col_1:
							st.image(recommended_movie_posters[0])
							
						with col_2:
							st.header(recommended_movie_names[0])
							st.subheader(main_movie_data['tagline'])
							st.write("Released On :{}".format(main_movie_data['release_date']))
							for current_genre in all_genres:
								tags_genre(current_genre)
							# st.markdown("""<p>Time:{}min</p>""".format(main_movie_data['runtime']))
							
							st.markdown("""<p><i class="fa fa-clock-o" aria-hidden="true">{}min</i></p>""".format(main_movie_data['runtime']),unsafe_allow_html=True)
							# st.markdown(f"""<div class="progress">
							# <div class="progress-bar bg-info" role="progressbar" style="width: {main_movie_data['popularity']}" aria-valuenow="{main_movie_data['popularity']}" aria-valuemin="0" aria-valuemax="100">
							#   {main_movie_data['popularity']}</div>
							# </div>""")
							st.markdown(main_movie_data['overview'])
							st_player('https://www.youtube.com/watch?v={}'.format(movie_video_key))

						col1, col2, col3, col4, col5 = st.columns(5)

						with col1:
							st.text(recommended_movie_names[1])
							st.image(recommended_movie_posters[1])
						
						with col2:
							st.text(recommended_movie_names[2])
							st.image(recommended_movie_posters[2])

						with col3:
							st.text(recommended_movie_names[3])
							st.image(recommended_movie_posters[3])
						
						with col4:
							st.text(recommended_movie_names[4])
							st.image(recommended_movie_posters[4])

						with col5:
							st.text(recommended_movie_names[5])
							st.image(recommended_movie_posters[5])

				elif choice == "Genre Based Recommendation":
					st.header('Movie Recommender System')

					movies['genres']=movies['genres'].apply(convert)

					genre_list = ['Action','Fantasy','Crime','Drama','Animation']

					selected_movie = st.selectbox(
						"Type or select a movie from the dropdown",
						genre_list
					)

					if st.button('Show Recommendation'):
						x=0
						recommended_movie_names = []
						recommended_movie_posters = []
						for i in range (0,100):
							if selected_movie in movies['genres'][i]:
								recommended_movie_names.append(credits['title'][i])
								recommended_movie_posters.append(fetch_poster(credits['movie_id'][i]))
								x=x+1
								if x>4:
									break
						col1, col2, col3, col4, col5 = st.columns(5)
						with col1:
							st.text(recommended_movie_names[0])
							st.image(recommended_movie_posters[0])
								
						with col2:
							st.text(recommended_movie_names[1])
							st.image(recommended_movie_posters[1])
							

						with col3:
							st.text(recommended_movie_names[2])
							st.image(recommended_movie_posters[2])
							
						with col4:
							st.text(recommended_movie_names[3])
							st.image(recommended_movie_posters[3])
							
						with col5:
							st.text(recommended_movie_names[4])
							st.image(recommended_movie_posters[4])

				elif choice == "Song Recommendation":
					title = "Rec-Ent Song"
					st.title(title)
					st.write("Welcome! This is the place where you can customize what you want to listen to based on genre and several key audio features. Try playing around with different settings and listen to the songs recommended by our system!")
					st.markdown("##")

					with st.container():
						col1, col2,col3,col4 = st.columns((2,0.5,0.5,0.5))
						with col3:
							st.markdown("***Choose your genre:***")
							genre = st.radio(
								"",
								genre_names, index=genre_names.index("Pop"))
						with col1:
							st.markdown("***Choose features to customize:***")
							start_year, end_year = st.slider(
								'Select the year range',
								1990, 2019, (2015, 2019)
							)
							acousticness = st.slider(
								'Acousticness',
								0.0, 1.0, 0.5)
							danceability = st.slider(
								'Danceability',
								0.0, 1.0, 0.5)
							energy = st.slider(
								'Energy',
								0.0, 1.0, 0.5)
							instrumentalness = st.slider(
								'Instrumentalness',
								0.0, 1.0, 0.0)
							valence = st.slider(
								'Valence',
								0.0, 1.0, 0.45)
							tempo = st.slider(
								'Tempo',
								0.0, 244.0, 118.0)

					tracks_per_page = 6
					test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
					uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
					tracks = []

					for uri in uris:
						track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
						tracks.append(track)

					if 'previous_inputs' not in st.session_state:
						st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat
					current_inputs = [genre, start_year, end_year] + test_feat

					if current_inputs != st.session_state['previous_inputs']:
						if 'start_track_i' in st.session_state:
							st.session_state['start_track_i'] = 0
						st.session_state['previous_inputs'] = current_inputs

					if 'start_track_i' not in st.session_state:
						st.session_state['start_track_i'] = 0

					with st.container():
						col1, col2, col3 = st.columns([2,1,2])
						if st.button("Recommend More Songs"):
							if st.session_state['start_track_i'] < len(tracks):
								st.session_state['start_track_i'] += tracks_per_page

						current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
						current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
						if st.session_state['start_track_i'] < len(tracks):
							for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
								if i%2==0:
									with col1:
										components.html(
										track,
											height=400,
										)
										with st.expander("See more details"):
											df = pd.DataFrame(dict(
											r=audio[:5],
											theta=audio_feats[:5]))
											fig = px.line_polar(df, r='r', theta='theta', line_close=True)
											fig.update_layout(height=400, width=340)
											st.plotly_chart(fig)
							
								else:
									with col3:
										components.html(
											track,
											height=400,
										)
										with st.expander("See more details"):
											df = pd.DataFrame(dict(
												r=audio[:5],
												theta=audio_feats[:5]))
											fig = px.line_polar(df, r='r', theta='theta', line_close=True)
											fig.update_layout(height=400, width=340)
											st.plotly_chart(fig)

						else:
							st.write("No songs left to recommend")

				elif choice == "Rate Rec-Ent":
					st.title("Customer Feedback")
					st.subheader(" ")
					# st.subheader("Thank you for your valuable review!")
					with st.form(key='nlpForm'):
						raw_text = st.text_area("Enter Text Here")
						submit_button = st.form_submit_button(label='Analyze')

					# layout
					col1,col2 = st.columns(2)
					if submit_button:
						
						with col1:
							# st.info("Results")
							sentiment = TextBlob(raw_text).sentiment
							# st.write(sentiment)

							# Emoji
							if sentiment.polarity > 0:
								st.subheader("Thank you for your valuable review!")
								st.markdown("Sentiment detected is Positive :smiley: ")
							elif sentiment.polarity < 0:
								st.subheader("Sorry for disappointing you.We will try to correct our mistakes soon!")
								st.markdown("Sentiment detected is Negative :angry: ")
							else:
								st.subheader("Thank you for your review!")
								st.markdown("Sentiment detected is Neutral 😐 ")

							# Dataframe
							result_df = convert_to_df(sentiment)
							# st.dataframe(result_df)

							# Visualization
							c = alt.Chart(result_df).mark_bar().encode(
								x='metric',
								y='value',
								color='metric')
							# st.altair_chart(c,use_container_width=True)



						with col2:
							# st.info("Token Sentiment")

							token_sentiments = analyze_token_sentiment(raw_text)
							# st.write(token_sentiments)

			else:
				st.warning("Incorrect Username/Password")


	elif choice == "SignUp":
		page_bg_img = '''
		<style>
      	.stApp {
  		background-image: url("https://payload.cargocollective.com/1/11/367710/13568488/MOVIECLASSICSerikweb_2500_800.jpg");
  		background-size: cover;
		}
		</style>'''

		# st.markdown(page_bg_img, unsafe_allow_html=True)
		st.markdown("<style>h1 {font-size: 70px;}<style/>",unsafe_allow_html=True)
		st.markdown("<h1>Rec-Ent Sign-Up<h1/>",unsafe_allow_html=True)
		st.markdown("<style>p{font-size: 30px;}<style/>",unsafe_allow_html=True)
		# st.markdown('<p>Login to our Rec-Ent for more movie and song recommendations<p/>',unsafe_allow_html=True)

		# st.subheader("Rec Ent Sign Up ")
		st.markdown("<p>Don't have an account yet? Create New Account<p/>",unsafe_allow_html=True)
		st.markdown('<p>Sign up our Rec-Ent Website <p/>',unsafe_allow_html=True)

		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("You can now login !!")

		
if __name__ == '__main__':
	main()
