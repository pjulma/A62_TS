mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"pierrot25fr@yahoo.fr\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
