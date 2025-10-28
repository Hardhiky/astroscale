open Lwt.Infix
open Cohttp
open Cohttp_lwt_unix

let port = 8080

let () = Printf.printf "Starting server on http://localhost:%d\n%!" port

let predict_handler _body =
  let open Yojson.Basic.Util in
  Cohttp_lwt.Body.to_string _body >>= fun body ->
  let json = Yojson.Basic.from_string body in
  let ra = json |> member "ra" |> to_float in
  let dec = json |> member "dec" |> to_float in
  let teff = json |> member "teff" |> to_float in
  let logg = json |> member "logg" |> to_float in
  let fe_h = json |> member "fe_h" |> to_float in
  let snr = json |> member "snr" |> to_float in
  let parallax = json |> member "parallax" |> to_float in

  let cmd =
    Printf.sprintf
      "cd .. && python3 inference.py %.6f %.6f %.6f %.6f %.6f %.6f %.6f"
      ra dec teff logg fe_h snr parallax
  in
  let ic = Unix.open_process_in cmd in
  let output = input_line ic in
  ignore (Unix.close_process_in ic);

  let resp =
    `Assoc [ ("predicted_z", `String output) ]
    |> Yojson.Basic.to_string
  in
  Server.respond_string ~status:`OK ~body:resp ()
;;

let server =
  let callback _conn req body =
    let path = Uri.path (Request.uri req) in
    let meth = Request.meth req in
    Printf.printf "[%s] %s\n%!" (Code.string_of_method meth) path;
    match path with
    | "/predict" -> predict_handler body
    | _ -> Server.respond_not_found ()
  in
  Server.create ~mode:(`TCP (`Port port)) (Server.make ~callback ())
;;

let () = Lwt_main.run server
