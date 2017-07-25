DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../libs

echo ${INSTALL_DIR}

# Execution environment variables
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${INSTALL_DIR}/bin:$PATH"

# Compilation environment variables
export C_INCLUDE_PATH="${INSTALL_DIR}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${INSTALL_DIR}/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="${INSTALL_DIR}/lib:$LIBRARY_PATH"
